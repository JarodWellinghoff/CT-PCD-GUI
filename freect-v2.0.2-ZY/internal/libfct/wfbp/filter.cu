#include <stdio.h>
#include <cuda.h>

extern "C" {
    void filter(float* projections, float* filtered_projections) {
        printf("rebin called\n");
        
        //int to_print = 10;
        //
        //printf("Some projection data:\n");
        //for (int i=0; i<to_print; i++) {
        //    printf("  %.4f\n", projections[i]);
        //}
        //
        //printf("Some table position data:\n");
        //for (int i=0; i<to_print; i++) {
        //    printf("  %.4f\n", projections[i]);
        //}
        //
        //printf("Some tube angle data:\n");
        //for (int i=0;i<to_print; i++){
        //    printf("  %.4f\n", projections[i]);
        //}
    }
}