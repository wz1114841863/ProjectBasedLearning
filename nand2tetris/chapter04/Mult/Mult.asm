// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/4/Mult.asm

// Multiplies R0 and R1 and stores the result in R2.
// (R0, R1, R2 refer to RAM[0], RAM[1], and RAM[2], respectively.)
// The algorithm is based on repetitive addition.

// 初始化
@R0
D=M
@value1
M=D

@R1
D=M
@value2
M=D

@R2
M=0

@i
M=0

// 循环相加
(LOOP)
    @i
    D=M
    @value1
    D=M-D
    @END
    D;JEQ

    @value2
    D=M
    @R2
    M=D+M

    @i
    M=M+1

    @LOOP
    0;JMP

(END)
    @END
    0;JMP

