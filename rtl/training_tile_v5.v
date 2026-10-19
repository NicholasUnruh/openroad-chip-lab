module training_tile_v5 (
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
    // Shift-add keeps the structural gate's multiplier count at the engine's eight lanes.
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
    // Next job to load and start: pair of output tiles p and q, inner tile k; bank = second.
    reg s_valid, second;
    reg [2:0] p_i, q_i;
    reg [1:0] p_j, q_j, s_k;
    wire q_valid = q_i <= {1'b0, m_max};
    wire [1:0] cur_i = second ? q_i[1:0] : p_i[1:0];
    wire [1:0] cur_j = second ? q_j : p_j;
    wire [2:0] after_q_i = q_j == n_max ? q_i + 1'b1 : q_i;
    wire [1:0] after_q_j = q_j == n_max ? 2'd0 : q_j + 1'b1;
    wire [2:0] after_p_i = p_j == n_max ? p_i + 1'b1 : p_i;
    wire [1:0] after_p_j = p_j == n_max ? 2'd0 : p_j + 1'b1;

    reg load_active, wr_valid, wr_b, loaded;
    reg [5:0] load_index;
    reg [5:0] wr_addr;
    reg [1:0] wr_lane;
    wire phase = load_index[5];
    wire [1:0] tile_row = phase ? (tb ? cur_j : s_k) : (ta ? s_k : cur_i);
    wire [1:0] tile_col = phase ? (tb ? s_k : cur_j) : (ta ? cur_i : s_k);
    wire [1:0] col_tiles = phase ? (tb ? k_max : n_max) : (ta ? m_max : k_max);
    wire [7:0] sp_row = (phase ? b_row0 : a_row0) +
                        {1'b0, scale({tile_row, load_index[4:2]}, col_tiles)} + {6'b0, tile_col};

    wire [127:0] sp_dout;
    wire [127:0] unused_dout0;
    sky130_sram_1rw1r_128x256_8 scratchpad (
        .clk0(clk), .csb0(!sp_write), .web0(1'b0), .wmask0(16'hf << {sp_addr[1:0], 2'b0}),
        .addr0(sp_addr[9:2]), .din0({4{sp_data}}), .dout0(unused_dout0),
        .clk1(clk), .csb1(!load_active), .addr1(sp_row), .dout1(sp_dout)
    );

    reg run_bank, run_final;
    reg [3:0] run_tile;
    reg [1:0] pending;
    reg oldest;
    reg [3:0] drain_tile0, drain_tile1;
    wire eng_busy, eng_done;
    wire [4:0] eng_flags;
    wire loading = load_active || wr_valid;
    // Operand loading owns the host bank; a drain of the other buffer waits until loading ends.
    wire host_bank = loading || !(|pending) ? second : oldest;
    assign result_valid = |pending && host_bank == oldest;
    assign result_tile = oldest ? drain_tile1 : drain_tile0;
    wire pop = result_ack && result_valid;
    wire finishing = eng_done && run_final;
    // Loading waits for computation on its bank, so a finishing job never shares the next job's bank.
    wire eng_start = gemm_busy && s_valid && loaded && !eng_busy && !pending[second];
    wire load_go = gemm_busy && s_valid && !loaded && !loading && !(eng_busy && run_bank == second);
    wire [1:0] push = finishing ? 2'b01 << run_bank : 2'b00;
    wire [1:0] pending_next = (pending & ~(pop ? 2'b01 << oldest : 2'b00)) | push;

    training_tile_engine #(.BUFFERED(1), .BF16_INPUT(1), .DIM_LOG2(3)) engine (
        .clk(clk), .rst(rst), .start(eng_start),
        .transpose_a(ta), .transpose_b(tb), .accumulate(s_k != 2'd0),
        .job_bank(second), .host_bank(host_bank),
        .write_enable(wr_valid ? (wr_b ? 3'b010 : 3'b001) : 3'b000),
        .write_addr(wr_addr), .write_data(sp_dout[{wr_lane, 5'b0} +: 32]),
        .read_addr(read_addr), .read_data(read_data), .write_ready(),
        .busy(eng_busy), .done(eng_done), .exception_flags(eng_flags)
    );

    always @(posedge clk) begin
        if (rst) begin
            gemm_busy <= 1'b0;
            gemm_done <= 1'b0;
            exception_flags <= 5'b0;
            s_valid <= 1'b0;
            second <= 1'b0;
            load_active <= 1'b0;
            wr_valid <= 1'b0;
            loaded <= 1'b0;
            pending <= 2'b0;
            oldest <= 1'b0;
            run_bank <= 1'b0;
            run_final <= 1'b0;
        end else begin
            gemm_done <= 1'b0;
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
                    s_valid <= 1'b1;
                    second <= 1'b0;
                    s_k <= 2'd0;
                    p_i <= 3'd0;
                    p_j <= 2'd0;
                    q_i <= n_tiles == 2'd0 ? 3'd1 : 3'd0;
                    q_j <= n_tiles == 2'd0 ? 2'd0 : 2'd1;
                    exception_flags <= 5'b0;
                end
            end else if (!s_valid && !eng_busy && !eng_done && !(|pending)) begin
                gemm_busy <= 1'b0;
                gemm_done <= 1'b1;
            end

            wr_valid <= load_active;
            wr_b <= phase;
            wr_addr <= {load_index[4:0], 1'b0};
            wr_lane <= load_index[1:0];
            if (load_go) begin
                load_active <= 1'b1;
                load_index <= 6'd0;
            end else if (load_active) begin
                load_index <= load_index + 1'b1;
                if (&load_index) load_active <= 1'b0;
            end
            if (wr_valid && !load_active) loaded <= 1'b1;

            if (eng_done) exception_flags <= exception_flags | eng_flags;
            if (finishing && !run_bank) drain_tile0 <= run_tile;
            if (finishing && run_bank) drain_tile1 <= run_tile;
            pending <= pending_next;
            if (pending_next == 2'b01 || pending_next == 2'b10) oldest <= pending_next[1];

            if (eng_start) begin
                loaded <= 1'b0;
                run_bank <= second;
                run_final <= s_k == k_max;
                run_tile <= {cur_i, cur_j};
                if (!second && q_valid) second <= 1'b1;
                else begin
                    second <= 1'b0;
                    if (s_k != k_max) s_k <= s_k + 1'b1;
                    else begin
                        s_k <= 2'd0;
                        p_i <= second ? after_q_i : after_p_i;
                        p_j <= second ? after_q_j : after_p_j;
                        q_i <= second ? (after_q_j == n_max ? after_q_i + 1'b1 : after_q_i) :
                                        (after_p_j == n_max ? after_p_i + 1'b1 : after_p_i);
                        q_j <= second ? (after_q_j == n_max ? 2'd0 : after_q_j + 1'b1) :
                                        (after_p_j == n_max ? 2'd0 : after_p_j + 1'b1);
                        if ((second ? after_q_i : after_p_i) > {1'b0, m_max}) s_valid <= 1'b0;
                    end
                end
            end
        end
    end
endmodule
