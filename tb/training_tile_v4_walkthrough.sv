module training_tile_v4_walkthrough;
    logic clk = 0;
    logic rst = 1;
    logic start = 0;
    logic transpose_a = 0;
    logic transpose_b = 0;
    logic accumulate = 0;
    logic job_bank = 0;
    logic host_bank = 0;
    logic [2:0] write_enable = 0;
    logic [5:0] write_addr = 0;
    logic [31:0] write_data = 0;
    logic [5:0] read_addr = 0;
    wire [31:0] read_data;
    wire write_ready, busy, done;
    wire [4:0] exception_flags;

    training_tile_v4 dut (.*);
    always #5 clk = ~clk;

    task automatic write_pair(input logic [2:0] enable, input logic [5:0] addr,
                              input logic [31:0] data);
        @(negedge clk);
        write_enable = enable;
        write_addr = addr;
        write_data = data;
        #1;
        if (!write_ready) $fatal(1, "write rejected at address %0d", addr);
    endtask

    initial begin
        $dumpfile("build/training-v4-walkthrough.vcd");
        $dumpvars(0, dut);

        repeat (2) @(negedge clk);
        rst = 0;
        for (int i = 0; i < 64; i += 2) write_pair(3'b001, 6'(i), 32'h3f80_3f80);
        for (int i = 0; i < 64; i += 2) write_pair(3'b010, 6'(i), 32'h3f80_3f80);

        @(negedge clk);
        write_enable = 0;
        start = 1;
        @(posedge clk);
        #1;
        if (!busy) $fatal(1, "job did not start");
        @(negedge clk);
        start = 0;

        repeat (67) @(posedge clk);
        #1;
        if (!done || busy || exception_flags != 0)
            $fatal(1, "unexpected completion: done=%b busy=%b flags=%b",
                   done, busy, exception_flags);
        for (int i = 0; i < 64; ++i) begin
            read_addr = 6'(i);
            #1;
            if (read_data !== 32'h4100_0000)
                $fatal(1, "C[%0d]=%h, expected 8.0", i, read_data);
        end
        $display("PASS: V4 8x8 ones matrix product, 67-cycle job, 64 FP32 outputs of 8.0");
        $finish;
    end
endmodule
