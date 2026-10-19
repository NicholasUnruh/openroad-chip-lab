// Scratchpad GEMM with four fused BF16 dot-product units (docs/training-v6.md); V5 host interface.
module training_tile_v6 (
    input wire clk, rst,
    input wire sp_write,
    input wire [9:0] sp_addr,
    input wire [31:0] sp_data,
    input wire gemm_start,
    input wire [1:0] m_tiles, n_tiles, k_tiles,
    input wire [7:0] a_base, b_base,
    input wire transpose_a, transpose_b,
    output reg gemm_busy, gemm_done,
    output reg [4:0] exception_flags,
    output wire result_valid,
    output wire [3:0] result_tile,
    input wire [5:0] read_addr,
    output wire [31:0] read_data,
    input wire result_ack
);
    // Shift-add keeps the structural gate's multiplier count at the dot-product units' 32.
    function [6:0] scale;
        input [4:0] row;
        input [1:0] tiles_minus_one;
        case (tiles_minus_one)
            2'd0: scale = {2'b0, row};
            2'd1: scale = {1'b0, row, 1'b0};
            2'd2: scale = {1'b0, row, 1'b0} + {2'b0, row};
            default: scale = {row, 2'b0};
        endcase
    endfunction

    reg [1:0] m_max, n_max, k_max;
    reg [7:0] a_row0, b_row0;
    reg ta, tb;

    // Loader job pointer: tiles row-major, K innermost; each tile alternates C banks.
    reg ld_valid, ld_buf, ld_cbank;
    reg [1:0] ld_i, ld_j, ld_k;
    // Job being loaded, and the loaded job waiting to compute.
    reg [1:0] li, lj, lk;
    reg nxt_valid, nxt_buf, nxt_cbank, nxt_k0, nxt_last;
    reg [3:0] nxt_tile;
    reg load_active, load_buf, loaded, wr_valid, wr_a, wr_buf;
    reg [3:0] load_index;
    reg [2:0] wr_row;

    // Eight B rows, then eight A rows; A row i is first read 2i clocks into its job.
    wire phase_a = load_index[3];
    wire [1:0] tile_row = phase_a ? (ta ? lk : li) : (tb ? lj : lk);
    wire [1:0] tile_col = phase_a ? (ta ? li : lk) : (tb ? lk : lj);
    wire [1:0] col_tiles = phase_a ? (ta ? m_max : k_max) : (tb ? k_max : n_max);
    wire [7:0] sp_row = (phase_a ? a_row0 : b_row0) +
                        {1'b0, scale({tile_row, load_index[2:0]}, col_tiles)} + {6'b0, tile_col};

    wire [127:0] sp_dout;
    wire [127:0] unused_dout0;
    sky130_sram_1rw1r_128x256_8 scratchpad (
        .clk0(clk), .csb0(!sp_write), .web0(1'b0), .wmask0(16'hf << {sp_addr[1:0], 2'b0}),
        .addr0(sp_addr[9:2]), .din0({4{sp_data}}), .dout0(unused_dout0),
        .clk1(clk), .csb1(!load_active), .addr1(sp_row), .dout1(sp_dout)
    );

    // Operands: a_reg[buf][i][k] and b_reg[buf][k][j], 16 bits each, row-major. All register
    // file accesses use constant indices with explicit selects; variable part-selects of these
    // wide vectors synthesize to barrel shifters.
    reg [2047:0] a_reg, b_reg;
    integer bi, x, y, u, k, e;
    always @(posedge clk)
        for (bi = 0; bi < 2; bi = bi + 1)
            for (x = 0; x < 8; x = x + 1)
                for (y = 0; y < 8; y = y + 1) begin
                    if (wr_valid && wr_a && wr_buf == bi[0] && (ta ? y[2:0] == wr_row : x[2:0] == wr_row))
                        a_reg[((bi*8+x)*8+y)*16 +: 16] <= ta ? sp_dout[16*x +: 16] : sp_dout[16*y +: 16];
                    if (wr_valid && !wr_a && wr_buf == bi[0] && (tb ? y[2:0] == wr_row : x[2:0] == wr_row))
                        b_reg[((bi*8+x)*8+y)*16 +: 16] <= tb ? sp_dout[16*x +: 16] : sp_dout[16*y +: 16];
                end

    reg run_active, run_buf, run_cbank, run_k0, run_last;
    reg [3:0] run_tile, run_cycle;
    wire [2:0] issue_row = run_cycle[3:1];
    wire issue_half = run_cycle[0];
    // Pipeline control matching the four-clock dot-product latency.
    reg [3:0] p_valid, p_last, p_cbank, p_half;
    reg [11:0] p_row;
    reg [15:0] p_tile;

    reg [1:0] pending;
    reg oldest;
    reg [3:0] drain_tile0, drain_tile1;
    assign result_valid = |pending;
    assign result_tile = oldest ? drain_tile1 : drain_tile0;
    wire pop = result_ack && result_valid;
    wire [1:0] push = p_valid[3] && p_last[3] ? 2'b01 << p_cbank[3] : 2'b00;
    wire [1:0] pending_next = (pending & ~(pop ? 2'b01 << oldest : 2'b00)) | push;

    // Accumulators c_reg[unit][{bank, row, half}]: unit u owns columns u and u+4.
    reg [4095:0] c_reg;
    reg [31:0] host_word;
    always @* begin
        host_word = 32'd0;
        for (u = 0; u < 4; u = u + 1)
            for (e = 0; e < 32; e = e + 1)
                if ({25'd0, read_addr[1:0], oldest, read_addr[5:3], read_addr[2]} == u*32 + e)
                    host_word = c_reg[(u*32+e)*32 +: 32];
    end
    assign read_data = host_word;

    // Untransposed A row i lands nine rows into the load and is needed 2i clocks into the job;
    // transposed A fills columns, so every row must land first.
    wire nxt_ready = loaded || (load_active ? !ta && load_index >= 4'd9 : wr_valid);
    wire slot_free = !run_active || run_cycle == 4'd15;
    // A later tile reuses a bank only after an intervening 16-clock job, past its last writeback.
    wire eng_start = nxt_valid && nxt_ready && slot_free && !(nxt_k0 && pending[nxt_cbank]);
    // Loads run back to back: the next job's first read follows the previous job's last read.
    // Buffers alternate and only one loaded job waits, so the running job never uses ld_buf.
    wire load_go = gemm_busy && ld_valid && !nxt_valid && (!load_active || &load_index);

    reg [127:0] a_row;
    reg [511:0] unit_b;
    reg [127:0] unit_c;
    wire [127:0] unit_result;
    wire [19:0] unit_flags;
    always @* begin
        a_row = 128'd0;
        for (bi = 0; bi < 2; bi = bi + 1)
            for (x = 0; x < 8; x = x + 1)
                if (run_buf == bi[0] && issue_row == x[2:0]) a_row = a_reg[(bi*8+x)*128 +: 128];
        unit_b = 512'd0;
        unit_c = 128'd0;
        for (u = 0; u < 4; u = u + 1) begin
            for (bi = 0; bi < 2; bi = bi + 1)
                for (k = 0; k < 8; k = k + 1)
                    if (run_buf == bi[0])
                        unit_b[128*u+16*k +: 16] = issue_half ? b_reg[((bi*8+k)*8+u+4)*16 +: 16]
                                                              : b_reg[((bi*8+k)*8+u)*16 +: 16];
            for (e = 0; e < 32; e = e + 1)
                if (!run_k0 && {run_cbank, issue_row, issue_half} == e[4:0])
                    unit_c[32*u +: 32] = c_reg[(u*32+e)*32 +: 32];
        end
    end
    genvar g;
    generate for (g = 0; g < 4; g = g + 1) begin: units
        bf16_dot8_fp32 dot (
            .clk(clk), .a(a_row), .b(unit_b[128*g +: 128]), .c(unit_c[32*g +: 32]),
            .result(unit_result[32*g +: 32]), .exception_flags(unit_flags[5*g +: 5])
        );
    end endgenerate

    always @(posedge clk) begin
        for (u = 0; u < 4; u = u + 1)
            for (e = 0; e < 32; e = e + 1)
                if (p_valid[3] && {p_cbank[3], p_row[11:9], p_half[3]} == e[4:0])
                    c_reg[(u*32+e)*32 +: 32] <= unit_result[32*u +: 32];
        p_row <= {p_row[8:0], issue_row};
        p_tile <= {p_tile[11:0], run_tile};
        p_last <= {p_last[2:0], run_last && run_cycle == 4'd15};
        p_cbank <= {p_cbank[2:0], run_cbank};
        p_half <= {p_half[2:0], issue_half};
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
            p_valid <= 4'b0;
            pending <= 2'b0;
            oldest <= 1'b0;
        end else begin
            gemm_done <= 1'b0;
            p_valid <= {p_valid[2:0], run_active};
            if (!gemm_busy) begin
                if (gemm_start) begin
                    gemm_busy <= 1'b1;
                    m_max <= m_tiles;
                    n_max <= n_tiles;
                    k_max <= k_tiles;
                    a_row0 <= a_base;
                    b_row0 <= b_base;
                    ta <= transpose_a;
                    tb <= transpose_b;
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
            wr_a <= phase_a;
            wr_row <= load_index[2:0];
            wr_buf <= load_buf;
            if (load_go) begin
                load_active <= 1'b1;
                load_index <= 4'd0;
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
                if (&load_index) load_active <= 1'b0;
            end
            if (wr_valid && !load_active && !load_go) loaded <= 1'b1;

            if (run_active) run_cycle <= run_cycle + 1'b1;
            if (run_active && run_cycle == 4'd15) run_active <= 1'b0;
            if (eng_start) begin
                nxt_valid <= 1'b0;
                run_active <= 1'b1;
                run_cycle <= 4'd0;
                run_buf <= nxt_buf;
                run_cbank <= nxt_cbank;
                run_k0 <= nxt_k0;
                run_last <= nxt_last;
                run_tile <= nxt_tile;
            end

            if (p_valid[3])
                exception_flags <= exception_flags | unit_flags[4:0] | unit_flags[9:5] |
                                   unit_flags[14:10] | unit_flags[19:15];
            if (push[0]) drain_tile0 <= p_tile[15:12];
            if (push[1]) drain_tile1 <= p_tile[15:12];
            pending <= pending_next;
            if (pending_next == 2'b01 || pending_next == 2'b10) oldest <= pending_next[1];
        end
    end
endmodule
