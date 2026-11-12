#include "../gpu_error_check.h"
#include <stdio.h>
#include <cuda.h>

texture<float,cudaTextureType2D,cudaReadModeElementType> tex_row_sheet;

//__global__ void rebin_kernel(float * output){
//    int channel_idx = threadIdx.x + blockDim.x*blockIdx.x;
//    int proj_idx    = threadIdx.y + blockDim.y*blockIdx.y;
//
//    float delta_theta = 2.0f*3.14159265f/(float)d_cg.projections_per_rotation;
//  
//    float p = ((float)channel_idx - d_cg.detector_central_col)*(float)d_cg.detector_pixel_size_col;
//    float beta = asin(p/d_cg.distance_source_to_detector);
//    float theta = (float)proj_idx * delta_theta;
//    float alpha = theta - beta;
//  
//    float beta_idx  = (beta*d_cg.distance_source_to_detector/d_cg.detector_pixel_size_col) + d_cg.detector_central_col;
//    float alpha_idx = alpha/delta_theta;
//  
//    // ORG Implementation
//    //int out_idx = channel_idx + proj_idx*d_cg.num_detector_cols;
//    //output[out_idx] = tex2D(tex_row_sheet, beta_idx + 0.5f, alpha_idx + 0.5);
//  
//    int channel_offset = (d_cg.num_detector_cols_padded_fft - d_cg.num_detector_cols)/2;
//    int out_idx = (channel_idx+channel_offset)  + proj_idx*(d_cg.num_detector_cols_padded_fft);
//
//    if (beta_idx < 0 || beta_idx >= d_cg.num_detector_cols || isnan(beta_idx)){
//        output[out_idx] = 0.0f;
//    }
//    else{
//        output[out_idx] = tex2D(tex_row_sheet, beta_idx + 0.5f, alpha_idx + 0.5);
//    }
//}

extern "C" {
    void rebin(float* projections, double* table_positions, double* tube_angles, float * rebinned_projections) {
        // vars to pass:
        // 

        printf("rebin called\n");
        
        int to_print = 10;
        
        printf("Some projection data:\n");
        for (int i=0; i<to_print; i++) {
            printf("  %.4f\n", projections[i]);
        }
        
        printf("Some table position data:\n");
        for (int i=0; i<to_print; i++) {
            printf("  %.4f\n", table_positions[i]);
        }
        
        printf("Some tube angle data:\n");
        for (int i=0;i<to_print; i++){
            printf("  %.9f\n", tube_angles[i]);
        }

        //////// Reshape the raw data to be "row sheets"
        //////// Additionally, flip the channel direction and row direction
        //////// since Chen et al 2015 defines the geometry to be the opposite
        //////// of how Stierstorfer et al 2004 defines it.
        //////// (May be worth it to eventually do this on the GPU...)        
        ////////std::cout << "Reshaping raw data array..." << std::endl;
        //////std::shared_ptr<float> raw_reshaped_ptr(new float[m_cg.num_detector_cols*m_cg.num_detector_rows*m_cg.total_number_of_projections]);
        //////float * raw = m_raw_data.get();
        //////float * raw_reshaped = raw_reshaped_ptr.get();
        //////for (int i=0; i<m_cg.total_number_of_projections;i++){
        //////    for (int j=0; j<m_cg.num_detector_rows;j++){
        //////        for (int k=0; k<m_cg.num_detector_cols;k++){
        //////            int input_idx = k + j*m_cg.num_detector_cols + i*m_cg.num_detector_cols*m_cg.num_detector_rows;
        //////            int output_idx = (m_cg.num_detector_cols - 1 - k) + i*m_cg.num_detector_cols + (m_cg.num_detector_rows - 1 - j)*m_cg.num_detector_cols*m_cg.total_number_of_projections;
        //////            //int output_idx = (m_cg.num_detector_cols - 1 - k) + i*m_cg.num_detector_cols + j*m_cg.num_detector_cols*m_cg.total_number_of_projections;
        //////            raw_reshaped[output_idx] = raw[input_idx];
        //////        }
        //////    }
        //////}
        ////////std::cout << "Done!" << std::endl;
        //////
        //////m_cg.detector_central_col = m_cg.detector_central_col;


        
        
    }
}