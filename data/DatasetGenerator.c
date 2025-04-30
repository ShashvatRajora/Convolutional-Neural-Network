#include <stdio.h>

extern unsigned char image[80][1024];  // your existing 80 images

unsigned char new_image[240][1024];

void create_synthetic_data() {
    // Copy original images
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 1024; j++) {
            new_image[i][j] = image[i][j];
        }
    }

    // Horizontally flipped images
    for (int i = 0; i < 80; i++) {
        for (int row = 0; row < 32; row++) {
            for (int col = 0; col < 32; col++) {
                int idx1 = row * 32 + col;
                int idx2 = row * 32 + (31 - col);
                new_image[i + 80][idx1] = image[i][idx2];
            }
        }
    }

    // Brightness increased images
    for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 1024; j++) {
            int val = image[i][j] + 30; // increase brightness
            if (val > 255) val = 255;
            new_image[i + 160][j] = (unsigned char)val;
        }
    }
}

void write_to_file() {
    FILE *f = fopen("data_face80.h", "w");
    if (!f) {
        printf("Error writing file!\n");
        return;
    }

    fprintf(f, "unsigned char image[240][1024] = {\n");

    for (int i = 0; i < 240; i++) {
        fprintf(f, " { ");
        for (int j = 0; j < 1024; j++) {
            fprintf(f, "%d", new_image[i][j]);
            if (j < 1023) fprintf(f, ", ");
        }
        fprintf(f, " },\n");
    }

    fprintf(f, "};\n");
    fclose(f);
    printf("New data_face80.h created!\n");
}

int main() {
    create_synthetic_data();
    write_to_file();
    return 0;
}
