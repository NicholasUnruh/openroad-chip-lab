`timescale 1ns/1ps

module mac8_tb;
    reg clk = 0;
    reg rst = 1;
    reg en = 0;
    reg [7:0] a = 0;
    reg [7:0] b = 0;
    wire [23:0] acc;
    integer i;

    mac8 dut (.clk(clk), .rst(rst), .en(en), .a(a), .b(b), .acc(acc));
    always #5 clk = ~clk;

    task expect_acc;
        input [23:0] expected;
        begin
            @(posedge clk);
            #1;
            if (acc !== expected)
                $fatal(1, "acc=%0d, expected=%0d", acc, expected);
        end
    endtask

    initial begin
        expect_acc(0);

        @(negedge clk);
        rst = 0;
        a = 3;
        b = 7;
        expect_acc(0);

        @(negedge clk);
        en = 1;
        expect_acc(21);

        @(negedge clk);
        a = 8'd255;
        b = 8'd255;
        expect_acc(65046);

        @(negedge clk);
        en = 0;
        expect_acc(65046);

        @(negedge clk);
        rst = 1;
        en = 1;
        expect_acc(0);

        @(negedge clk);
        rst = 0;
        for (i = 1; i <= 259; i = i + 1)
            expect_acc(i * 65025);

        $display("PASS: mac8 reset priority, enable, multiply, accumulate, hold, wrap");
        $finish;
    end
endmodule
