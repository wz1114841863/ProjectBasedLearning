// This file is part of www.nand2tetris.org
// and the book "The Elements of Computing Systems"
// by Nisan and Schocken, MIT Press.
// File name: projects/4/Fill.asm

// Runs an infinite loop that listens to the keyboard input. 
// When a key is pressed (any key), the program blackens the screen,
// i.e. writes "black" in every pixel. When no key is pressed, 
// the screen should be cleared.

@value1
M=-1  // -1用二进制补码表示，意味着将寄存器所有位置置1
      // 进而在后面的程序可以用来将一个字的像素点亮

@value0
M=0   // 用于熄灭一个字的像素

(LOOP)
      @24576
      D=M  // 获得键盘输入对应的ASK码

      @POSITIVE
      D;JNE  // ASK码不为零，代码有输入，跳转

      @IFZERO
      D;JEQ

(POSITIVE)
      @value1
      D=M
      @16384
      M=D
      
      @LOOP
      0;JMP

(IFZERO)
      @value0
      D=M
      @16384
      M=D

      @LOOP
      0;JMP





