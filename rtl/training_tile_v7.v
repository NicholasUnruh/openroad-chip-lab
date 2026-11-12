// Scratchpad GEMM with two MXFP8 32-term block dot-product units (docs/training-v7.md).
// The V6 sequencer and scratchpad, with 8x8x32 jobs: FP8 halves the bytes per element, so the
// same 128-bit read port feeds 64 multiply-accumulates per clock instead of 32.
module training_tile_v7 (
    input wire clk, rst,
    input wire sp_write,
    input wire [9:0] sp_addr,
    input wire [31:0] sp_data,
    input wire gemm_start,
    input wire [1:0] m_tiles, n_tiles, k_tiles,
    input wire [7:0] a_base, b_base, a_scale_base, b_scale_base,
    output reg gemm_busy, gemm_done,
    output reg [4:0] exception_flags,
    output wire result_valid,
    output wire [3:0] result_tile,
    input wire [5:0] read_addr,
    output wire [31:0] read_data,
    input wire result_ack
);
    // value * (k_tiles + 1) by shift-add, so the only multipliers are the dot-product units'.
    function [6:0] times_blocks;
        input [4:0] value;
        input [1:0] blocks_minus_one;
        case (blocks_minus_one)
            2'd0: times_blocks = {2'b0, value};
            2'd1: times_blocks = {1'b0, value, 1'b0};
            2'd2: times_blocks = {1'b0, value, 1'b0} + {2'b0, value};
            default: times_blocks = {value, 2'b0};
        endcase
    endfunction

    reg [1:0] m_max, n_max, k_max;
    reg [7:0] a_row0, b_row0, a_scale_row0, b_scale_row0;

    // Loader job pointer: tiles row-major, K blocks innermost; each tile alternates C banks.
    reg ld_valid, ld_buf, ld_cbank;
    reg [1:0] ld_i, ld_j, ld_k;
    // Job being loaded, and the loaded job waiting to compute.
    reg [1:0] li, lj, lk;
    reg nxt_valid, nxt_buf, nxt_cbank, nxt_k0, nxt_last;
    reg [3:0] nxt_tile;
    reg load_active, load_buf, loaded, wr_valid, wr_buf;
    reg [5:0] load_index;
    reg [4:0] wr_slot;
    reg wr_scale;

    // Load order, 34 reads per job: A scales, B scales, then 16 B rows and 16 A rows.
    // Slot {operand, row, half}: operand 0 = B, 1 = A; half selects elements 0-15 or 16-31.
    wire load_scale = load_index < 6'd2;
    wire [4:0] slot = load_index[4:0] - 5'd2;
    wire slot_a = load_scale ? !load_index[0] : slot[4];
    wire [1:0] tile = slot_a ? li : lj;
    // A matrix row of K = 32 * (k_tiles + 1) bytes spans 2 * (k_tiles + 1) scratchpad rows;
    // the scale row of (tile, K block) is tile * (k_tiles + 1) + block.
    wire [7:0] scale_row = (slot_a ? a_scale_row0 : b_scale_row0) +
                           {1'b0, times_blocks({3'b0, tile}, k_max)} + {6'b0, lk};
    wire [7:0] data_row = (slot_a ? a_row0 : b_row0) +
                          {times_blocks({tile, slot[3:1]}, k_max), 1'b0} + {5'b0, lk, slot[0]};
    wire [7:0] sp_row = load_scale ? scale_row : data_row;

    wire [127:0] sp_dout;
    wire [127:0] unused_dout0;
    sky130_sram_1rw1r_128x256_8 scratchpad (
        .clk0(clk), .csb0(!sp_write), .web0(1'b0), .wmask0(16'hf << {sp_addr[1:0], 2'b0}),
        .addr0(sp_addr[9:2]), .din0({4{sp_data}}), .dout0(unused_dout0),
        .clk1(clk), .csb1(!load_active), .addr1(sp_row), .dout1(sp_dout)
    );

    // Operands: a_reg[buf][row][k] and b_reg[buf][col][k], one E4M3 byte each, K-contiguous;
    // scales sa_reg[buf][row] and sb_reg[buf][col], one E8M0 byte each. All accesses use
    // constant indices with explicit selects; variable part-selects of these wide vectors
    // synthesize to barrel shifters.
    reg [4095:0] a_reg, b_reg;
    reg [127:0] sa_reg, sb_reg;
    integer bi, x, h, u, e;
    always @(posedge clk)
        for (bi = 0; bi < 2; bi = bi + 1) begin
            if (wr_valid && wr_buf == bi[0] && wr_scale && wr_slot[4])
                sa_reg[64*bi +: 64] <= sp_dout[63:0];
            if (wr_valid && wr_buf == bi[0] && wr_scale && !wr_slot[4])
                sb_reg[64*bi +: 64] <= sp_dout[63:0];
            for (x = 0; x < 8; x = x + 1)
                for (h = 0; h < 2; h = h + 1) begin
                    if (wr_valid && wr_buf == bi[0] && !wr_scale && wr_slot == {1'b1, x[2:0], h[0]})
                        a_reg[(bi*8+x)*256+128*h +: 128] <= sp_dout;
                    if (wr_valid && wr_buf == bi[0] && !wr_scale && wr_slot == {1'b0, x[2:0], h[0]})
                        b_reg[(bi*8+x)*256+128*h +: 128] <= sp_dout;
                end
        end

    // Each clock of a 32-clock job broadcasts A row issue_row; unit u computes column
    // 4u + issue_col. Results write back LAT clocks later.
    reg run_active, run_buf, run_cbank, run_k0, run_last;
    reg [3:0] run_tile;
    reg [4:0] run_cycle;
    wire [2:0] issue_row = run_cycle[4:2];
    wire [1:0] issue_col = run_cycle[1:0];
    localparam LAT = 7;
    reg [LAT-1:0] p_valid, p_last, p_cbank;
    reg [3*LAT-1:0] p_row;
    reg [2*LAT-1:0] p_col;
    reg [4*LAT-1:0] p_tile;

    reg [1:0] pending;
    reg oldest;
    reg [3:0] drain_tile0, drain_tile1;
    assign result_valid = |pending;
    assign result_tile = oldest ? drain_tile1 : drain_tile0;
    wire pop = result_ack && result_valid;
    wire [1:0] push = p_valid[LAT-1] && p_last[LAT-1] ? 2'b01 << p_cbank[LAT-1] : 2'b00;
    wire [1:0] pending_next = (pending & ~(pop ? 2'b01 << oldest : 2'b00)) | push;

    // Accumulators c_reg[unit][{bank, row, col}]: unit u owns columns 4u to 4u+3.
    reg [4095:0] c_reg;
    reg [31:0] host_word;
    always @* begin
        host_word = 32'd0;
        for (u = 0; u < 2; u = u + 1)
            for (e = 0; e < 64; e = e + 1)
                if ({25'd0, read_addr[2], oldest, read_addr[5:3], read_addr[1:0]} == u*64 + e)
                    host_word = c_reg[(u*64+e)*32 +: 32];
    end
    assign read_data = host_word;

    // A row r is complete two clocks after its second read (load index 19 + 2r) and is first
    // needed 4r clocks into the job, so the job may start at load index 20.
    wire nxt_ready = loaded || (load_active ? load_index >= 6'd20 : wr_valid);
    wire slot_free = !run_active || run_cycle == 5'd31;
    // A later tile reuses a bank only after an intervening 32-clock job, past its last writeback.
    wire eng_start = nxt_valid && nxt_ready && slot_free && !(nxt_k0 && pending[nxt_cbank]);
    // Loads run back to back: the next job's first read follows the previous job's last read.
    // Buffers alternate and only one loaded job waits, so the running job never uses ld_buf.
    wire load_go = gemm_busy && ld_valid && !nxt_valid && (!load_active || load_index == 6'd33);

    reg [255:0] a_row;
    reg [511:0] unit_b;
    reg [7:0] row_scale;
    reg [15:0] unit_scale;
    reg [63:0] unit_c;
    wire [63:0] unit_result;
    wire [9:0] unit_flags;
    always @* begin
        a_row = 256'd0;
        row_scale = 8'd0;
        for (bi = 0; bi < 2; bi = bi + 1)
            for (x = 0; x < 8; x = x + 1)
                if (run_buf == bi[0] && issue_row == x[2:0]) begin
                    a_row = a_reg[(bi*8+x)*256 +: 256];
                    row_scale = sa_reg[64*bi+8*x +: 8];
                end
        unit_b = 512'd0;
        unit_scale = 16'd0;
        unit_c = 64'd0;
        for (u = 0; u < 2; u = u + 1) begin
            for (bi = 0; bi < 2; bi = bi + 1)
                for (x = 0; x < 4; x = x + 1)
                    if (run_buf == bi[0] && issue_col == x[1:0]) begin
                        unit_b[256*u +: 256] = b_reg[(bi*8+4*u+x)*256 +: 256];
                        unit_scale[8*u +: 8] = sb_reg[64*bi+8*(4*u+x) +: 8];
                    end
            for (e = 0; e < 64; e = e + 1)
                if (!run_k0 && {run_cbank, issue_row, issue_col} == e[5:0])
                    unit_c[32*u +: 32] = c_reg[(u*64+e)*32 +: 32];
        end
    end
    genvar g;
    generate for (g = 0; g < 2; g = g + 1) begin: dot_units
        mx_dot32_fp32 #(.ELEM(0)) dot (
            .clk(clk), .a(a_row), .b(unit_b[256*g +: 256]), .scale_a(row_scale),
            .scale_b(unit_scale[8*g +: 8]), .c(unit_c[32*g +: 32]),
            .result(unit_result[32*g +: 32]), .exception_flags(unit_flags[5*g +: 5])
        );
    end endgenerate

    always @(posedge clk) begin
        for (u = 0; u < 2; u = u + 1)
            for (e = 0; e < 64; e = e + 1)
                if (p_valid[LAT-1] && {p_cbank[LAT-1], p_row[3*LAT-1 -: 3], p_col[2*LAT-1 -: 2]} == e[5:0])
                    c_reg[(u*64+e)*32 +: 32] <= unit_result[32*u +: 32];
        p_row <= {p_row[3*LAT-4:0], issue_row};
        p_col <= {p_col[2*LAT-3:0], issue_col};
        p_tile <= {p_tile[4*LAT-5:0], run_tile};
        p_last <= {p_last[LAT-2:0], run_last && run_cycle == 5'd31};
        p_cbank <= {p_cbank[LAT-2:0], run_cbank};
        if (rst) begin
            gemm_busy <= 1'b0;
            gemm_done <= 1'b0;
            exception_flags <= 5'b0;
            ld_valid <= 1'b0;
            nxt_valid <= 1'b0;
            load_active <= 1'b0;
            loaded <= 1'b0;
            wr_valid <= 1'b0;
            run_active <= 1'b0;
            p_valid <= {LAT{1'b0}};
            pending <= 2'b0;
            oldest <= 1'b0;
        end else begin
            gemm_done <= 1'b0;
            p_valid <= {p_valid[LAT-2:0], run_active};
            if (!gemm_busy) begin
                if (gemm_start) begin
                    gemm_busy <= 1'b1;
                    m_max <= m_tiles;
                    n_max <= n_tiles;
                    k_max <= k_tiles;
                    a_row0 <= a_base;
                    b_row0 <= b_base;
                    a_scale_row0 <= a_scale_base;
                    b_scale_row0 <= b_scale_base;
                    ld_valid <= 1'b1;
                    ld_buf <= 1'b0;
                    ld_cbank <= 1'b0;
                    ld_i <= 2'd0;
                    ld_j <= 2'd0;
                    ld_k <= 2'd0;
                    exception_flags <= 5'b0;
                end
            end else if (!ld_valid && !nxt_valid && !run_active && !(|p_valid) && !(|pending)) begin
                gemm_busy <= 1'b0;
                gemm_done <= 1'b1;
            end

            wr_valid <= load_active;
            wr_scale <= load_scale;
            wr_slot <= load_scale ? {!load_index[0], 4'd0} : slot;
            wr_buf <= load_buf;
            if (load_go) begin
                load_active <= 1'b1;
                load_index <= 6'd0;
                loaded <= 1'b0;
                load_buf <= ld_buf;
                li <= ld_i;
                lj <= ld_j;
                lk <= ld_k;
                nxt_valid <= 1'b1;
                nxt_buf <= ld_buf;
                nxt_cbank <= ld_cbank;
                nxt_k0 <= ld_k == 2'd0;
                nxt_last <= ld_k == k_max;
                nxt_tile <= {ld_i, ld_j};
                ld_buf <= !ld_buf;
                if (ld_k != k_max) ld_k <= ld_k + 1'b1;
                else begin
                    ld_k <= 2'd0;
                    ld_cbank <= !ld_cbank;
                    ld_j <= ld_j == n_max ? 2'd0 : ld_j + 1'b1;
                    if (ld_j == n_max) begin
                        ld_i <= ld_i + 1'b1;
                        if (ld_i == m_max) ld_valid <= 1'b0;
                    end
                end
            end else if (load_active) begin
                load_index <= load_index + 1'b1;
                if (load_index == 6'd33) load_active <= 1'b0;
            end
            if (wr_valid && !load_active && !load_go) loaded <= 1'b1;

            if (run_active) run_cycle <= run_cycle + 1'b1;
            if (run_active && run_cycle == 5'd31) run_active <= 1'b0;
            if (eng_start) begin
                nxt_valid <= 1'b0;
                run_active <= 1'b1;
                run_cycle <= 5'd0;
                run_buf <= nxt_buf;
                run_cbank <= nxt_cbank;
                run_k0 <= nxt_k0;
                run_last <= nxt_last;
                run_tile <= nxt_tile;
            end

            if (p_valid[LAT-1]) exception_flags <= exception_flags | unit_flags[4:0] | unit_flags[9:5];
            if (push[0]) drain_tile0 <= p_tile[4*LAT-1 -: 4];
            if (push[1]) drain_tile1 <= p_tile[4*LAT-1 -: 4];
            pending <= pending_next;
            if (pending_next == 2'b01 || pending_next == 2'b10) oldest <= pending_next[1];
        end
    end
endmodule
