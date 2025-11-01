/**
 * @file test_rl_primitives.c
 * @brief Test program for RL primitives
 *
 * This program tests the low-level DRAM command primitives
 * without requiring actual hardware.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Mock structures (simplified versions)
typedef struct {
    uint32_t cap0_0_1 : 2;
    uint32_t cap0_2_5R0_3 : 4;
    uint32_t cap0_6_7BA0_1 : 2;
    uint32_t cap0_8_10BG0_2 : 3;
    uint32_t cap0_11_13CID0_2 : 3;
    uint32_t cap1_0_12R4_16 : 13;
    uint32_t cap1_13R17CID3 : 1;
} CMD_ACT;

typedef union {
    CMD_ACT act;
    // Other command types would go here
} CMD_TRUTH_TABLE;

// Test build_activate_cmd
int build_activate_cmd(uint8_t rank, uint8_t bg, uint8_t ba, uint32_t row,
                      CMD_TRUTH_TABLE *cmd_out) {
    if (!cmd_out) return -1;

    // Validate parameters
    if (rank > 3 || bg > 7 || ba > 3 || row > 262143) {
        printf("ERROR: Invalid ACT parameters: rank=%d, bg=%d, ba=%d, row=%d\n",
               rank, bg, ba, row);
        return -1;
    }

    memset(cmd_out, 0, sizeof(CMD_TRUTH_TABLE));

    // Set command type (ACT = 0x00)
    cmd_out->act.cap0_0_1 = 0x00;

    // Encode address
    cmd_out->act.cap0_2_5R0_3 = row & 0xF;              // Row[0:3]
    cmd_out->act.cap0_6_7BA0_1 = ba & 0x3;              // Bank[0:1]
    cmd_out->act.cap0_8_10BG0_2 = bg & 0x7;             // Bank Group[0:2]
    cmd_out->act.cap0_11_13CID0_2 = rank & 0x7;         // Rank/CID[0:2]
    cmd_out->act.cap1_0_12R4_16 = (row >> 4) & 0x1FFF;  // Row[4:16]
    cmd_out->act.cap1_13R17CID3 = (row >> 17) & 0x1;    // Row[17]

    printf("Built ACT command - rank=%d, bg=%d, ba=%d, row=%d\n",
           rank, bg, ba, row);

    return 0;
}

void print_cmd_act(CMD_ACT *act) {
    printf("  cap0_0_1: 0x%x\n", act->cap0_0_1);
    printf("  cap0_2_5R0_3: 0x%x (row[0:3])\n", act->cap0_2_5R0_3);
    printf("  cap0_6_7BA0_1: 0x%x (bank)\n", act->cap0_6_7BA0_1);
    printf("  cap0_8_10BG0_2: 0x%x (bank group)\n", act->cap0_8_10BG0_2);
    printf("  cap0_11_13CID0_2: 0x%x (rank[0:2])\n", act->cap0_11_13CID0_2);
    printf("  cap1_0_12R4_16: 0x%x (row[4:16])\n", act->cap1_0_12R4_16);
    printf("  cap1_13R17CID3: 0x%x (row[17])\n", act->cap1_13R17CID3);
}

int main() {
    CMD_TRUTH_TABLE cmd;
    int result;

    printf("=== RL Primitives Test Program ===\n\n");

    // Test 1: Simple ACTIVATE
    printf("Test 1: ACTIVATE rank=0, bg=0, ba=0, row=100\n");
    result = build_activate_cmd(0, 0, 0, 100, &cmd);
    if (result == 0) {
        print_cmd_act(&cmd.act);
        printf("✓ Test 1 PASSED\n\n");
    } else {
        printf("✗ Test 1 FAILED\n\n");
    }

    // Test 2: ACTIVATE with different parameters
    printf("Test 2: ACTIVATE rank=1, bg=3, ba=2, row=12345\n");
    result = build_activate_cmd(1, 3, 2, 12345, &cmd);
    if (result == 0) {
        print_cmd_act(&cmd.act);
        printf("✓ Test 2 PASSED\n\n");
    } else {
        printf("✗ Test 2 FAILED\n\n");
    }

    // Test 3: Large row address
    printf("Test 3: ACTIVATE rank=2, bg=7, ba=3, row=262000\n");
    result = build_activate_cmd(2, 7, 3, 262000, &cmd);
    if (result == 0) {
        print_cmd_act(&cmd.act);
        printf("✓ Test 3 PASSED\n\n");
    } else {
        printf("✗ Test 3 FAILED\n\n");
    }

    // Test 4: Invalid parameters (should fail)
    printf("Test 4: ACTIVATE with invalid row (300000)\n");
    result = build_activate_cmd(0, 0, 0, 300000, &cmd);
    if (result != 0) {
        printf("✓ Test 4 PASSED (correctly rejected)\n\n");
    } else {
        printf("✗ Test 4 FAILED (should have rejected)\n\n");
    }

    printf("=== All Tests Complete ===\n");

    return 0;
}
