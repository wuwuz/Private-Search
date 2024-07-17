#include "textflag.h"

// Function signature: func L2DistanceSIMD(a, b *float32, n int) float32
TEXT ·L2DistanceSIMD(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX              // Load address of a into AX
    MOVQ b+8(FP), BX              // Load address of b into BX
    MOVQ n+16(FP), CX             // Load length n into CX (number of floats, not chunks)

    VXORPS Y0, Y0, Y0             // Initialize Y0 to zero for accumulating sums

    XORQ DX, DX                   // Initialize loop counter DX to 0
    LOOP:
        VMOVUPS (AX)(DX*4), Y1    // Load 8 floats from a into Y1
        VMOVUPS (BX)(DX*4), Y2    // Load 8 floats from b into Y2

        VSUBPS Y2, Y1, Y1         // Y1 = Y1 - Y2
        VMULPS Y1, Y1, Y1         // Y1 = Y1 * Y1 (square differences)

        VADDPS Y1, Y0, Y0         // Y0 += Y1 (accumulate sums)

        ADDQ $8, DX               // Increment loop counter by 8 (processing 8 floats at a time)
        CMPQ DX, CX               // Compare loop counter with length
        JL LOOP                   // If counter < length, continue loop

    // Extract upper 128 bits to X1
    VEXTRACTF128 $1, Y0, X1

    // Add upper 128 bits to lower 128 bits
    VHADDPS X1, X0, X0 // VADDPS or VHADDPS?

    // Horizontal add of the 4 floats in X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    MOVSS X0, ret+24(FP)          // Move the result to return value slot
    RET

