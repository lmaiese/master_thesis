#include "gasal.h"
#include "args_parser.h"
#include "res.h"
#include "gasal_align.h"
#include "gasal_kernels.h"

#include <omp.h>
//#include "adept_utils.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include "libStatGen/include/SamFile.h"
#include "libStatGen/include/SamFileHeader.h"


inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band, data_source semiglobal_skipping_head, data_source semiglobal_skipping_tail, Bool secondBest) 
{
	switch(algo)
	{
		
		KERNEL_SWITCH(LOCAL,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(SEMI_GLOBAL,  start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);		// MACRO that expands all 32 semi-global kernels
		KERNEL_SWITCH(GLOBAL,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(KSW,			start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(BANDED,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		default:
		break;

	}

}


//GASAL2 asynchronous (a.k.a non-blocking) alignment function
void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params) {

	cudaError_t err;
	if (actual_n_alns <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_n_alns <= 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_query_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes <= 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes <= 0\n");
		exit(EXIT_FAILURE);
	}

	if (actual_query_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes=%d is not a multiple of 8\n", actual_query_batch_bytes);
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes=%d is not a multiple of 8\n", actual_target_batch_bytes);
		exit(EXIT_FAILURE);
	}

	if (actual_query_batch_bytes > gpu_storage->host_max_query_batch_bytes) {
				fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes(%d) > host_max_query_batch_bytes(%d)\n", actual_query_batch_bytes, gpu_storage->host_max_query_batch_bytes);
				exit(EXIT_FAILURE);
	}

	if (actual_target_batch_bytes > gpu_storage->host_max_target_batch_bytes) {
			fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes(%d) > host_max_target_batch_bytes(%d)\n", actual_target_batch_bytes, gpu_storage->host_max_target_batch_bytes);
			exit(EXIT_FAILURE);
	}

	if (actual_n_alns > gpu_storage->host_max_n_alns) {
			fprintf(stderr, "[GASAL ERROR:] actual_n_alns(%d) > host_max_n_alns(%d)\n", actual_n_alns, gpu_storage->host_max_n_alns);
			exit(EXIT_FAILURE);
	}

	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage->gpu_max_query_batch_bytes < actual_query_batch_bytes) {

		int i = 2;
		while ( (gpu_storage->gpu_max_query_batch_bytes * i) < actual_query_batch_bytes) i++;

		fprintf(stderr, "[GASAL WARNING:] actual_query_batch_bytes(%d) > Allocated GPU memory (gpu_max_query_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_query_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_query_batch_bytes, gpu_storage->gpu_max_query_batch_bytes, gpu_storage->gpu_max_query_batch_bytes*i, gpu_storage->gpu_max_query_batch_bytes*i);

		gpu_storage->gpu_max_query_batch_bytes = gpu_storage->gpu_max_query_batch_bytes * i;

		if (gpu_storage->unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_query_batch));
		if (gpu_storage->packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_query_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_query_batch), gpu_storage->gpu_max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_query_batch), (gpu_storage->gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));

	}

	if (gpu_storage->gpu_max_target_batch_bytes < actual_target_batch_bytes) {

		int i = 2;
		while ( (gpu_storage->gpu_max_target_batch_bytes * i) < actual_target_batch_bytes) i++;
		
		fprintf(stderr, "[GASAL WARNING:] actual_target_batch_bytes(%d) > Allocated GPU memory (gpu_max_target_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_target_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_target_batch_bytes, gpu_storage->gpu_max_target_batch_bytes, gpu_storage->gpu_max_target_batch_bytes*i, gpu_storage->gpu_max_target_batch_bytes*i);

		gpu_storage->gpu_max_target_batch_bytes = gpu_storage->gpu_max_target_batch_bytes * i;

		if (gpu_storage->unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_target_batch));
		if (gpu_storage->packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_target_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_target_batch), gpu_storage->gpu_max_target_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_target_batch), (gpu_storage->gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->gpu_max_n_alns < actual_n_alns) {

		int i = 2;
		while ( (gpu_storage->gpu_max_n_alns * i) < actual_n_alns) i++;
		
		fprintf(stderr, "[GASAL WARNING:] actual_n_alns(%d) > gpu_max_n_alns(%d). Therefore, allocating memory for %d alignments on  GPU (gpu_max_n_alns=%d). Performance may be lost if this is repeated many times.\n", actual_n_alns, gpu_storage->gpu_max_n_alns, gpu_storage->gpu_max_n_alns*i, gpu_storage->gpu_max_n_alns*i);

		gpu_storage->gpu_max_n_alns = gpu_storage->gpu_max_n_alns * i;

		if (gpu_storage->query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_offsets));
		if (gpu_storage->target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_offsets));
		if (gpu_storage->query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_lens));
		if (gpu_storage->target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_lens));

		if (gpu_storage->seed_scores != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->seed_scores));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_lens), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_lens), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_offsets), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_offsets), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->seed_scores), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		gasal_res_destroy_device(gpu_storage->device_res, gpu_storage->device_cpy);
		gpu_storage->device_cpy = gasal_res_new_device_cpy(gpu_storage->gpu_max_n_alns, params);
		gpu_storage->device_res = gasal_res_new_device(gpu_storage->device_cpy);

		if (params->secondBest)
		{
			gasal_res_destroy_device(gpu_storage->device_res_second, gpu_storage->device_cpy_second);
			gpu_storage->device_cpy_second = gasal_res_new_device_cpy(gpu_storage->gpu_max_n_alns, params);
			gpu_storage->device_res_second = gasal_res_new_device(gpu_storage->device_cpy_second);
		}

	}
	//------------------------------------------

	//------------------------launch copying of sequence batches from CPU to GPU---------------------------

	// here you can track the evolution of your data structure processing with the printer: gasal_host_batch_printall(current);

	host_batch_t *current = gpu_storage->extensible_host_unpacked_query_batch;
	while (current != NULL)
	{
		if (current->next != NULL ) 
		{
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_query_batch[current->offset]), 
											current->data, 
											current->next->offset - current->offset,
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
			
		} else {
			// it's the last page to copy
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_query_batch[current->offset]), 
											current->data, 
											actual_query_batch_bytes - current->offset, 
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
		}
		current = current->next;
	}

	current = gpu_storage->extensible_host_unpacked_target_batch;
	while (current != NULL)
	{
		if (current->next != NULL ) {
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_target_batch[current->offset]), 
											current->data, 
											current->next->offset - current->offset,
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );

		} else {
			// it's the last page to copy
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_target_batch[current->offset]), 
											current->data, 
											actual_target_batch_bytes - current->offset, 
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
		}
		current = current->next;
	}

	//-----------------------------------------------------------------------------------------------------------
	// TODO: Adjust the block size depending on the kernel execution.
	
    uint32_t BLOCKDIM = 64;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int query_batch_tasks_per_thread = (int)ceil((double)actual_query_batch_bytes/(8*BLOCKDIM*N_BLOCKS));
    int target_batch_tasks_per_thread = (int)ceil((double)actual_target_batch_bytes/(8*BLOCKDIM*N_BLOCKS));


    //-------------------------------------------launch packing kernel


	if (!(params->isPacked))
	{
		gasal_pack_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked_query_batch),
		(uint32_t*)(gpu_storage->unpacked_target_batch), gpu_storage->packed_query_batch, gpu_storage->packed_target_batch,
		query_batch_tasks_per_thread, target_batch_tasks_per_thread, actual_query_batch_bytes/4, actual_target_batch_bytes/4);
		cudaError_t pack_kernel_err = cudaGetLastError();
		if ( cudaSuccess != pack_kernel_err )
		{
		fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(pack_kernel_err), pack_kernel_err,  __LINE__, __FILE__);
		exit(EXIT_FAILURE);
		}
	}
    

	// We could reverse-complement before packing, but we would get 2x more read-writes to memory.

    //----------------------launch copying of sequence offsets and lengths from CPU to GPU--------------------------------------
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_batch_lens, gpu_storage->host_query_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_batch_lens, gpu_storage->host_target_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_batch_offsets, gpu_storage->host_query_batch_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_batch_offsets, gpu_storage->host_target_batch_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
	
	// if needed copy seed scores
	if (params->algo == KSW)
	{
		if (gpu_storage->seed_scores == NULL)
		{
			fprintf(stderr, "seed_scores == NULL\n");
			
		}
		if (gpu_storage->host_seed_scores == NULL)
		{
			fprintf(stderr, "host_seed_scores == NULL\n");
		}
		if (gpu_storage->seed_scores == NULL || gpu_storage->host_seed_scores == NULL)
			exit(EXIT_FAILURE);

		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->seed_scores, gpu_storage->host_seed_scores, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
	}
    //--------------------------------------------------------------------------------------------------------------------------

	//----------------------launch copying of sequence operations (reverse/complement) from CPU to GPU--------------------------
	if (params->isReverseComplement)
	{
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_op, gpu_storage->host_query_op, actual_n_alns * sizeof(uint8_t), cudaMemcpyHostToDevice,  gpu_storage->str));
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_op, gpu_storage->host_target_op, actual_n_alns * sizeof(uint8_t), cudaMemcpyHostToDevice,  gpu_storage->str));	
		//--------------------------------------launch reverse-complement kernel------------------------------------------------------
		gasal_reversecomplement_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
			gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->query_op, gpu_storage->target_op, actual_n_alns);
		cudaError_t reversecomplement_kernel_err = cudaGetLastError();
		if ( cudaSuccess != reversecomplement_kernel_err )
		{
			 fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(reversecomplement_kernel_err), reversecomplement_kernel_err,  __LINE__, __FILE__);
			 exit(EXIT_FAILURE);
		}
	
	}
	
    //--------------------------------------launch alignment kernels--------------------------------------------------------------
	gasal_kernel_launcher(N_BLOCKS, BLOCKDIM, params->algo, params->start_pos, gpu_storage, actual_n_alns, params->k_band, params->semiglobal_skipping_head, params->semiglobal_skipping_tail, params->secondBest);


        //-----------------------------------------------------------------------------------------------------------------------
    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(aln_kernel_err), aln_kernel_err,  __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    //------------------------0launch the copying of alignment results from GPU to CPU--------------------------------------
    if (gpu_storage->host_res->aln_score != NULL && gpu_storage->device_cpy->aln_score != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->aln_score, gpu_storage->device_cpy->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->query_batch_start != NULL && gpu_storage->device_cpy->query_batch_start != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->query_batch_start, gpu_storage->device_cpy->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->target_batch_start != NULL && gpu_storage->device_cpy->target_batch_start != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->target_batch_start, gpu_storage->device_cpy->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->query_batch_end != NULL && gpu_storage->device_cpy->query_batch_end != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->query_batch_end, gpu_storage->device_cpy->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->target_batch_end != NULL && gpu_storage->device_cpy->target_batch_end != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->target_batch_end, gpu_storage->device_cpy->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
	//-----------------------------------------------------------------------------------------------------------------------
	

	// not really needed to filter with params->secondBest, since all the pointers will be null and non-initialized.
	if (params->secondBest)
	{	
		if (gpu_storage->host_res_second->aln_score != NULL && gpu_storage->device_cpy_second->aln_score != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->aln_score, gpu_storage->device_cpy_second->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
		if (gpu_storage->host_res_second->query_batch_start != NULL && gpu_storage->device_cpy_second->query_batch_start != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->query_batch_start, gpu_storage->device_cpy_second->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->target_batch_start != NULL && gpu_storage->device_cpy_second->target_batch_start != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->target_batch_start, gpu_storage->device_cpy_second->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->query_batch_end != NULL && gpu_storage->device_cpy_second->query_batch_end != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->query_batch_end, gpu_storage->device_cpy_second->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->target_batch_end != NULL && gpu_storage->device_cpy_second->target_batch_end != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->target_batch_end, gpu_storage->device_cpy_second->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
	}

    gpu_storage->is_free = 0; //set the availability of current stream to false

}


int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage) 
{
	cudaError_t err;
	if(gpu_storage->is_free == 1) return -2;//if no work is launced in this stream, return -2
	err = cudaStreamQuery(gpu_storage->str);//check to see if the stream is finished
	if (err != cudaSuccess ) {
		if (err == cudaErrorNotReady) return -1;
		else{
			fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__);
			exit(EXIT_FAILURE);
		}
	}
	gpu_storage->is_free = 1;

	return 0;
}


void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = (subst->gap_open + subst->gap_extend);
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}

//NEW CODE

gpu_alignments::gpu_alignments(int max_alignments){
    cudaMalloc(&offset_query_gpu, (max_alignments) * sizeof(int));
    cudaMalloc(&offset_ref_gpu, (max_alignments) * sizeof(int));
    cudaMalloc(&ref_start_gpu, (max_alignments) * sizeof(short));
    cudaMalloc(&ref_end_gpu, (max_alignments) * sizeof(short));
    cudaMalloc(&query_end_gpu, (max_alignments) * sizeof(short));
    cudaMalloc(&query_start_gpu, (max_alignments) * sizeof(short));
    cudaMalloc(&scores_gpu, (max_alignments) * sizeof(short));
}


gpu_alignments::~gpu_alignments(){
    cudaFree(offset_ref_gpu);
    cudaFree(offset_query_gpu);
    cudaFree(ref_start_gpu);
    cudaFree(ref_end_gpu);
    cudaFree(query_start_gpu);
    cudaFree(query_end_gpu);
}


unsigned getMaxLength (std::vector<std::string> v)
{
  unsigned maxLength = 0;
  for(auto str : v){
    if(maxLength < str.length()){
      maxLength = str.length();
    }
  }
  return maxLength;
}


void initialize_alignments(alignment_results *alignments, int max_alignments){
    cudaMallocHost(&(alignments->ref_begin), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->ref_end), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->query_begin), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->query_end), sizeof(short)*max_alignments);
    cudaMallocHost(&(alignments->top_scores), sizeof(short)*max_alignments);
}


void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A,
unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda){

        cudaMemcpyAsync(gpu_data->offset_ref_gpu, offsetA_h, (sequences_per_stream) * sizeof(int),
        cudaMemcpyHostToDevice,streams_cuda[0]);
        cudaMemcpyAsync(gpu_data->offset_ref_gpu + sequences_per_stream, offsetA_h + sequences_per_stream,
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), cudaMemcpyHostToDevice,streams_cuda[1]);

        cudaMemcpyAsync(gpu_data->offset_query_gpu, offsetB_h, (sequences_per_stream) * sizeof(int),
        cudaMemcpyHostToDevice,streams_cuda[0]);
        cudaMemcpyAsync(gpu_data->offset_query_gpu + sequences_per_stream, offsetB_h + sequences_per_stream,
        (sequences_per_stream + sequences_stream_leftover) * sizeof(int), cudaMemcpyHostToDevice,streams_cuda[1]);

        cudaMemcpyAsync(strA_d, strA, half_length_A * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[0]);
        cudaMemcpyAsync(strA_d + half_length_A, strA + half_length_A, (totalLengthA - half_length_A) * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[1]);

        cudaMemcpyAsync(strB_d, strB, half_length_B * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[0]);
        cudaMemcpyAsync(strB_d + half_length_B, strB + half_length_B, (totalLengthB - half_length_B) * sizeof(char),
                              cudaMemcpyHostToDevice,streams_cuda[1]);
}


void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda){
            cudaMemcpyAsync(alAend, gpu_data->ref_end_gpu, sequences_per_stream * sizeof(short),
                cudaMemcpyDeviceToHost, streams_cuda[0]);
            cudaMemcpyAsync(alAend + sequences_per_stream, gpu_data->ref_end_gpu + sequences_per_stream,
                (sequences_per_stream + sequences_stream_leftover) * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[1]);

            cudaMemcpyAsync(alBend, gpu_data->query_end_gpu, sequences_per_stream * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[0]);
            cudaMemcpyAsync(alBend + sequences_per_stream, gpu_data->query_end_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                cudaMemcpyDeviceToHost, streams_cuda[1]);
}


void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda){
           cudaMemcpyAsync(alAbeg, gpu_data->ref_start_gpu, sequences_per_stream * sizeof(short),
                                  cudaMemcpyDeviceToHost, streams_cuda[0]);
           cudaMemcpyAsync(alAbeg + sequences_per_stream, gpu_data->ref_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                                  cudaMemcpyDeviceToHost, streams_cuda[1]);

          cudaMemcpyAsync(alBbeg, gpu_data->query_start_gpu, sequences_per_stream * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[0]);
          cudaMemcpyAsync(alBbeg + sequences_per_stream, gpu_data->query_start_gpu + sequences_per_stream, (sequences_per_stream + sequences_stream_leftover) * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[1]);

          cudaMemcpyAsync(top_scores_cpu, gpu_data->scores_gpu, sequences_per_stream * sizeof(short),
                          cudaMemcpyDeviceToHost, streams_cuda[0]);
          cudaMemcpyAsync(top_scores_cpu + sequences_per_stream, gpu_data->scores_gpu + sequences_per_stream,
          (sequences_per_stream + sequences_stream_leftover) * sizeof(short), cudaMemcpyDeviceToHost, streams_cuda[1]);
}


int get_new_min_length(short* alAend, short* alBend, int blocksLaunched){
        int newMin = 1000;
        int maxA = 0;
        int maxB = 0;
        for(int i = 0; i < blocksLaunched; i++){
          if(alBend[i] > maxB ){
              maxB = alBend[i];
          }
          if(alAend[i] > maxA){
            maxA = alAend[i];
          }
        }
        newMin = (maxB > maxA)? maxA : maxB;
        return newMin;
}


int LongestIn(std::vector<std::string> const &vec) 
{
    std::string sol;
    std::string::size_type max = 0;

    for(std::vector<std::string>::const_iterator itr = vec.begin(); itr != vec.end(); ++itr)
    {
        if(itr->size() > max)
        { 
            max = itr->size();
            sol = *itr;
        }
    }
    return sol.length();
}


void write_sam_and_stats(std::vector<std::string> refs, std::vector<std::string> quers, alignment_results *results) {

	std::cerr << "Writing Sam file..." << std::endl;

	SamFile samout;
	samout.OpenForWrite("./new_protein_alignments.sam");
	SamFileHeader samHeader;

	std::string hd = "@HD\tVN:1.0\tSO:queryname\n";
	size_t ref_len = 0;
	size_t ref_len_max = 0;
	size_t quer_len = 0;

	std::string::size_type max = 0;
	int k = 0;
	int j = 0;
	int tot_quers_len = 0;

	int quers_lens[quers.size()];

	for(std::vector<std::string>::const_iterator itr = quers.begin(); itr != quers.end(); ++itr)
        {
                quer_len = itr->size();
                tot_quers_len = tot_quers_len + quer_len;
		quers_lens[j] = quer_len;
		j++;
	}

	int avg_read_length = tot_quers_len/quers.size();

	for(std::vector<std::string>::const_iterator itr = refs.begin(); itr != refs.end(); ++itr)
	{
		ref_len = itr->size();
		if(ref_len > max)
			max = ref_len;
		std::string line = ("@SQ\tSN:" + refs[k] + "\tLN:" + std::to_string(static_cast<int>(ref_len)) + "\n");
		hd = hd + line;
		k++;
	}

	/*
	for(int i = 0; i<refs.size(); i++) {
		std::string *it = &refs[i];
		ref_len = (*it).length();
		if(ref_len > ref_len_max)
			ref_len_max = ref_len;
		std::string line = ("@SQ\tSN:" + refs[i] + "\tLN:" + std::to_string(static_cast<int>(ref_len)) + "\n");
		hd = hd + line;
	}
	*/

	hd = hd + "@PG\tID:Gasal2\tVN:1.0\n";
	const char* temp = hd.c_str();
	bool flag = samHeader.addHeader(temp);
        samout.WriteHeader(samHeader);

	//float min_score = -0.6 + (-0.6 * (ref_len_max/(refs.size() + quers.size())); //GLOBAL

	std::cerr << "REFS.SIZE(): " << refs.size() << std::endl;
	std::cerr << "QUERS.SIZE(): " << quers.size() << std::endl;

	//float argument = (max/(refs.size() + quers.size()));

	float argument = avg_read_length;

	std::cerr << "LOG ARGUMENT: " << std::to_string(argument) << std::endl;

	//float min_score = 20 + 8.0 * log(ref_len_max/(refs.size() + quers.size())); //LOCAL
	float min_score = 20 + 8.0 * log(argument); //LOCAL

	std::cerr << "\n" << "max: " << std::to_string(max) << "\n" << std::endl;
	std::cerr << "min_score: " << std::to_string(min_score) << "\n" << std::endl;
	std::cerr << "avg_read_length: " << std::to_string(avg_read_length) << "\n" << std::endl;
	std::cerr << "PROVAAAAAA" << "\n" << std::endl;

	for(int i = 0; i<quers.size(); i++) {
                SamRecord samRecord;
		std::string unavailable = "*";

                samRecord.setReadName(quers[i].c_str());
		float new_min_score = 20 + 8.0 * log(quers_lens[i]);
		std::cerr << "\n" << "NEW_MIN_SCORE: " << std::to_string(new_min_score) << "\n" << std::endl;
                if(results->top_scores[i] < min_score)
                        samRecord.setFlag(4);
		else
                        samRecord.setFlag(3);
                samRecord.setReferenceName(samHeader, refs[i].c_str());
		samRecord.set1BasedPosition(results->query_begin[i]);

		float diff = std::abs(min_score);
		float best_over = results->top_scores[i] - min_score;
		float best_diff = std::abs(std::abs(results->top_scores[i]));

			if(best_diff >= diff*0.9) {
				if(best_over == diff)
					samRecord.setMapQuality(39);
				else
					samRecord.setMapQuality(33);
			}
			else if(best_diff >= diff*0.8) {
				if(best_over == diff)
					samRecord.setMapQuality(38);
				else
					samRecord.setMapQuality(27);
			}
			else if(best_diff >= diff * 0.97) {
				if(best_over == diff)
					samRecord.setMapQuality(37);
				else
					samRecord.setMapQuality(26);
			}
			else if(best_diff >= diff*0.6) {
				if(best_over == diff)
					samRecord.setMapQuality(36);
				else
					samRecord.setMapQuality(22);
			}
			else if(best_diff >= diff*0.5) {
                                if(best_over == diff)
                                        samRecord.setMapQuality(35);
                                else if(best_over >= diff*0.84)
                                        samRecord.setMapQuality(25);
				else if(best_over >= diff*0.68)
					samRecord.setMapQuality(16);
				else
					samRecord.setMapQuality(5);
			}
			else if(best_diff >= diff*0.4) {
                                if(best_over == diff)
                                        samRecord.setMapQuality(34);
                                else if(best_over >= diff*0.84)
                                        samRecord.setMapQuality(21);
                                else if(best_over >= diff*0.68)
                                        samRecord.setMapQuality(14);
                                else
                                	 samRecord.setMapQuality(4);
			}
			else if(best_diff >= diff*0.3) {
                                if(best_over == diff)
                                        samRecord.setMapQuality(32);
                                else if(best_over >= diff*0.88)
                                        samRecord.setMapQuality(18);
                                else if(best_over >= diff*0.67)
                                        samRecord.setMapQuality(15);
                                else
                                        samRecord.setMapQuality(3);
			}
			else if(best_diff >= diff*0.2) {
                                if(best_over == diff)
                                        samRecord.setMapQuality(31);
                                else if(best_over >= diff*0.88)
                                        samRecord.setMapQuality(17);
                                else if(best_over >= diff*0.67)
                                        samRecord.setMapQuality(11);
                                else
                                        samRecord.setMapQuality(0);
			}
			else if(best_diff >= diff*0.1) {
                                if(best_over == diff)
                                        samRecord.setMapQuality(30);
                                else if(best_over >= diff*0.88)
                                        samRecord.setMapQuality(12);
                                else if(best_over >= diff*0.67)
                                        samRecord.setMapQuality(7);
                                else
                                        samRecord.setMapQuality(0);
			}
			else if(best_diff > 0) {
                                if(best_over >= diff*0.67)
                                        samRecord.setMapQuality(6);
                                else
                                        samRecord.setMapQuality(2);
			}
			else {
				samRecord.setMapQuality(255);
			}

		samRecord.setCigar(unavailable.c_str());
		samRecord.setMateReferenceName(samHeader, unavailable.c_str());
		samRecord.set1BasedMatePosition(0);
		samRecord.setInsertSize(results->query_end[i] - results->query_begin[i] + 1);
		samRecord.setSequence(quers[i].c_str());
		samRecord.setQuality(unavailable.c_str());
		samout.WriteRecord(samHeader, samRecord);
        }

	samout.Close();

	SamFile samin;
	samin.OpenForRead("./new_protein_alignments.sam");
	samin.GenerateStatistics(true);

	SamFileHeader samHeader2;
	samin.ReadHeader(samHeader2);
	SamRecord samRecord2;

	while(samin.ReadRecord(samHeader2, samRecord2)){}

	std::cerr << "Statistics:\n" << std::endl;
	samin.PrintStatistics();

	samin.Close();
}


void kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap) {
    std::cout<<"Proviamo a fare un debuggino:"<<"\n";
    unsigned maxContigSize = getMaxLength(contigs);
    unsigned maxReadSize = getMaxLength(reads);
    unsigned totalAlignments = contigs.size(); // assuming that read and contig vectors are same length
    short encoding_matrix[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             23,0,0,0,0,0,0,0,0,0,0,0,0,0,
                             0,0,0,0,0,0,0,0,0,0,20,4,3,6,
                             13,7,8,9,0,11,10,12,2,0,14,5,
                             1,15,16,0,19,17,22,18,21};
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    omp_set_num_threads(24);// one OMP thread per GPU
    std::cout<<"Number of available GPUs:"<<deviceCount<<"\n";
    cudaDeviceProp prop[deviceCount];
    for(int i = 0; i < deviceCount; i++)
      cudaGetDeviceProperties(&prop[i], 0);
    unsigned NBLOCKS = totalAlignments;
    unsigned alignmentsPerDevice = NBLOCKS / deviceCount;
    unsigned leftOver_device = NBLOCKS % deviceCount;
    int its = (totalAlignments>20000)?(ceil((float)totalAlignments/20000)):1;
    initialize_alignments(alignments, totalAlignments); // pinned memory allocation
    auto start = NOW;
    #pragma omp parallel
    {
      int my_cpu_id = omp_get_thread_num()%2;
      std::cerr << "\n" << "[INFO] my_cpu_id: " << my_cpu_id << "\n" << std::endl;
      cudaSetDevice(my_cpu_id);
      int myGPUid;
      cudaGetDevice(&myGPUid);
      float total_time_cpu = 0;
      cudaStream_t streams_cuda[NSTREAMS];
      for(int stm = 0; stm < NSTREAMS; stm++){
        cudaStreamCreate(&streams_cuda[stm]);
      }
      int BLOCKS_l = alignmentsPerDevice;
      if(my_cpu_id == deviceCount - 1)
          BLOCKS_l += leftOver_device;
      if(my_cpu_id == 0)std::cout<<"Number of GPUs being used:"<<omp_get_num_threads()<<"\n";
      unsigned leftOvers = BLOCKS_l % its;
      unsigned stringsPerIt = BLOCKS_l / its;
      gpu_alignments gpu_data(stringsPerIt + leftOvers); // gpu mallocs
      short *d_encoding_matrix, *d_scoring_matrix;
      cudaMalloc(&d_encoding_matrix, ENCOD_MAT_SIZE * sizeof(short));
      cudaMalloc(&d_scoring_matrix, SCORE_MAT_SIZE * sizeof(short));
      cudaMemcpy(d_encoding_matrix, encoding_matrix, ENCOD_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice);
      cudaMemcpy(d_scoring_matrix, scoring_matrix, SCORE_MAT_SIZE * sizeof(short), cudaMemcpyHostToDevice);
      short* alAbeg = alignments->ref_begin + my_cpu_id * alignmentsPerDevice;
      short* alBbeg = alignments->query_begin + my_cpu_id * alignmentsPerDevice;
      short* alAend = alignments->ref_end + my_cpu_id * alignmentsPerDevice;
      short* alBend = alignments->query_end + my_cpu_id * alignmentsPerDevice; // memory on CPU for copying the results
      short* top_scores_cpu = alignments->top_scores + my_cpu_id * alignmentsPerDevice;
      unsigned* offsetA_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetA_h, sizeof(int)*(stringsPerIt + leftOvers));
      unsigned* offsetB_h;// = new unsigned[stringsPerIt + leftOvers];
      cudaMallocHost(&offsetB_h, sizeof(int)*(stringsPerIt + leftOvers));
      char *strA_d, *strB_d;
      cudaMalloc(&strA_d, maxContigSize * (stringsPerIt + leftOvers) * sizeof(char));
      cudaMalloc(&strB_d, maxReadSize *(stringsPerIt + leftOvers)* sizeof(char));
      char* strA;
      cudaMallocHost(&strA, sizeof(char)*maxContigSize * (stringsPerIt + leftOvers));
      char* strB;
      cudaMallocHost(&strB, sizeof(char)* maxReadSize *(stringsPerIt + leftOvers));
      float total_packing = 0;
      auto start2 = NOW;

      std::cout<<"loop begingg\n";
      std::cout<<"mado quanto non mi è chiaro\n";


      for(int perGPUIts = 0; perGPUIts < its; perGPUIts++)
      {
          auto packing_start = NOW;
          int blocksLaunched = 0;
          std::vector<std::string>::const_iterator beginAVec;
          std::vector<std::string>::const_iterator endAVec;
          std::vector<std::string>::const_iterator beginBVec;
          std::vector<std::string>::const_iterator endBVec;
          if(perGPUIts == its - 1)
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt) + leftOvers;
              blocksLaunched = stringsPerIt + leftOvers;
          }
          else
          {
              beginAVec = contigs.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endAVec = contigs.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
              beginBVec = reads.begin() + ((alignmentsPerDevice * my_cpu_id) + perGPUIts * stringsPerIt);
              endBVec = reads.begin() + (alignmentsPerDevice * my_cpu_id) + (perGPUIts + 1) * stringsPerIt; // so that each openmp thread has a copy of strings it needs to align
              blocksLaunched = stringsPerIt;
          }
          std::vector<std::string> sequencesA(beginAVec, endAVec);
          std::vector<std::string> sequencesB(beginBVec, endBVec);
          unsigned running_sum = 0;
          int sequences_per_stream = (blocksLaunched) / NSTREAMS;
          int sequences_stream_leftover = (blocksLaunched) % NSTREAMS;
          unsigned half_length_A = 0;
          unsigned half_length_B = 0;
          auto start_cpu = NOW;
          for(int i = 0; i < sequencesA.size(); i++)
          {
              running_sum +=sequencesA[i].size();
              offsetA_h[i] = running_sum;//sequencesA[i].size();
              if(i == sequences_per_stream - 1){
                  half_length_A = running_sum;
                  running_sum = 0;
                }
          }
          unsigned totalLengthA = half_length_A + offsetA_h[sequencesA.size() - 1];
          running_sum = 0;
          for(int i = 0; i < sequencesB.size(); i++)
          {
              running_sum +=sequencesB[i].size();
              offsetB_h[i] = running_sum; //sequencesB[i].size();
              if(i == sequences_per_stream - 1){
                half_length_B = running_sum;
                running_sum = 0;
              }
          }
          unsigned totalLengthB = half_length_B + offsetB_h[sequencesB.size() - 1];
          auto end_cpu = NOW;
          std::chrono::duration<double> cpu_dur = end_cpu - start_cpu;
          total_time_cpu += cpu_dur.count();
          unsigned offsetSumA = 0;
          unsigned offsetSumB = 0;
          for(int i = 0; i < sequencesA.size(); i++)
          {
              char* seqptrA = strA + offsetSumA;
              memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
              char* seqptrB = strB + offsetSumB;
              memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
              offsetSumA += sequencesA[i].size();
              offsetSumB += sequencesB[i].size();
          }
          auto packing_end = NOW;
          std::chrono::duration<double> packing_dur = packing_end - packing_start;
          total_packing += packing_dur.count();
          asynch_mem_copies_htd(&gpu_data, offsetA_h, offsetB_h, strA, strA_d, strB, strB_d, half_length_A, half_length_B, totalLengthA, totalLengthB, sequences_per_stream, 
sequences_stream_leftover, streams_cuda);
          unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
          unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
          unsigned alignmentPad = 4 + (4 - totShmem % 4);
          size_t ShmemBytes = totShmem + alignmentPad;
          if(ShmemBytes > 48000)
              cudaFuncSetAttribute(sequence_dna_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ShmemBytes);
          sequence_aa_kernel<<<sequences_per_stream, minSize, ShmemBytes, streams_cuda[0]>>>(
              strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
              gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu,
              openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaGetLastError();
          sequence_aa_kernel<<<sequences_per_stream + sequences_stream_leftover, minSize, ShmemBytes, streams_cuda[1]>>>(
              strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream,
                gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu + 
sequences_per_stream,
                gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaGetLastError();
          // copyin back end index so that we can find new min
          asynch_mem_copies_dth_mid(&gpu_data, alAend, alBend, sequences_per_stream, sequences_stream_leftover, streams_cuda);
          cudaStreamSynchronize (streams_cuda[0]);
          cudaStreamSynchronize (streams_cuda[1]);
          auto sec_cpu_start = NOW;
          int newMin = get_new_min_length(alAend, alBend, blocksLaunched); // find the new largest of smaller lengths
          auto sec_cpu_end = NOW;
          std::chrono::duration<double> dur_sec_cpu = sec_cpu_end - sec_cpu_start;
          total_time_cpu += dur_sec_cpu.count();
          sequence_aa_reverse<<<sequences_per_stream, newMin, ShmemBytes, streams_cuda[0]>>>(
                  strA_d, strB_d, gpu_data.offset_ref_gpu, gpu_data.offset_query_gpu, gpu_data.ref_start_gpu,
                  gpu_data.ref_end_gpu, gpu_data.query_start_gpu, gpu_data.query_end_gpu, gpu_data.scores_gpu, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaGetLastError();
          sequence_aa_reverse<<<sequences_per_stream + sequences_stream_leftover, newMin, ShmemBytes, streams_cuda[1]>>>(
                  strA_d + half_length_A, strB_d + half_length_B, gpu_data.offset_ref_gpu + sequences_per_stream, gpu_data.offset_query_gpu + sequences_per_stream ,
                  gpu_data.ref_start_gpu + sequences_per_stream, gpu_data.ref_end_gpu + sequences_per_stream, gpu_data.query_start_gpu + sequences_per_stream, gpu_data.query_end_gpu 
+ sequences_per_stream,
                  gpu_data.scores_gpu + sequences_per_stream, openGap, extendGap, d_scoring_matrix, d_encoding_matrix);
          cudaGetLastError();
          asynch_mem_copies_dth(&gpu_data, alAbeg, alBbeg, top_scores_cpu, sequences_per_stream, sequences_stream_leftover, streams_cuda);
                alAbeg += stringsPerIt;
                alBbeg += stringsPerIt;
                alAend += stringsPerIt;
                alBend += stringsPerIt;
                top_scores_cpu += stringsPerIt;
      }  // for iterations end here
        auto end1 = NOW;
        std::chrono::duration<double> diff2 = end1 - start2;
        cudaFree(strA_d);
        cudaFree(strB_d);
        cudaFreeHost(offsetA_h);
        cudaFreeHost(offsetB_h);
        cudaFreeHost(strA);
        cudaFreeHost(strB);
        for(int i = 0; i < NSTREAMS; i++)
          cudaStreamDestroy(streams_cuda[i]);
        std::cout <<"cpu time:"<<total_time_cpu<<std::endl;
        std::cout <<"packing time:"<<total_packing<<std::endl;
        #pragma omp barrier
    }  // paralle pragma ends
    auto end = NOW;
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total Alignments:"<<totalAlignments<<"\n"<<"Max Reference Size:"<<maxContigSize<<"\n"<<"Max Query Size:"<<maxReadSize<<"\n" <<"Total Execution Time (seconds):"<< 
diff.count() <<std::endl;
}
