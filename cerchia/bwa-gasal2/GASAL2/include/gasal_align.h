#ifndef __GASAL_ALIGN_H__
#define __GASAL_ALIGN_H__
/*  ####################################################################################
    SEMI_GLOBAL Kernels generation - read from the bottom one, all the way up. (the most specialized ones are written before the ones that call them)
    ####################################################################################
*/
#define SEMIGLOBAL_KERNEL_CALL(a,s,h,t,b) \
	case t:\
		gasal_semi_global_kernel<Int2Type<a>, Int2Type<s>, Int2Type<b>, Int2Type<h>, Int2Type<t>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->device_res, gpu_storage->device_res_second, actual_n_alns); \
	break;

#define SWITCH_SEMI_GLOBAL_TAIL(a,s,h,t,b) \
	case h:\
	switch(t) { \
		SEMIGLOBAL_KERNEL_CALL(a,s,h,NONE,b)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,QUERY,b)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,TARGET,b)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,BOTH,b)\
	}\
	break;

#define SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t,b) \
	case s:\
	switch(h) { \
		SWITCH_SEMI_GLOBAL_TAIL(a,s,NONE,t,b)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,QUERY,t,b)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,TARGET,t,b)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,BOTH,t,b)\
	} \
	break;\


/*  ####################################################################################
    ALGORITHMS Kernels generation. Allows to have a single line written for all kernels calls. The switch-cases are MACRO-generated.
    #################################################################################### 
*/

#define SWITCH_SEMI_GLOBAL(a,s,h,t,b) SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t,b)

#define SWITCH_LOCAL(a,s,h,t,b) \
    case s:\
        gasal_local_kernel<Int2Type<LOCAL>, Int2Type<s>, Int2Type<b>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->device_res, gpu_storage->device_res_second, actual_n_alns); \
    break;

#define SWITCH_MICROLOCAL(a,s,h,t,b) \
    case s:\
        gasal_local_kernel<Int2Type<MICROLOCAL>, Int2Type<s>, Int2Type<b>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->device_res, gpu_storage->device_res_second, actual_n_alns);\
    break;

#define SWITCH_GLOBAL(a,s,h,t,b) \
    case s:\
        gasal_global_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->device_res, actual_n_alns);\
    break;


#define SWITCH_KSW(a,s,h,t,b) \
    case s:\
        gasal_ksw_kernel<Int2Type<b>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->seed_scores, gpu_storage->device_res, gpu_storage->device_res_second, actual_n_alns, zdrop);\
    break;

#define SWITCH_BANDED(a,s,h,t,b) \
    case s:\
        gasal_banded_tiled_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->device_res, actual_n_alns, k_band>>3); \
break;

/*  ####################################################################################
    RUN PARAMETERS calls : general call (bottom, should be used), and first level TRUE/FALSE calculation for second best, 
    then 2nd level WITH / WITHOUT_START switch call (top)
    ####################################################################################
*/

#define SWITCH_START(a,s,h,t,b) \
    case b: \
    switch(s){\
        SWITCH_## a(a,WITH_START,h,t,b)\
        SWITCH_## a(a,WITHOUT_START,h,t,b)\
    } \
    break;

#define SWITCH_SECONDBEST(a,s,h,t,b) \
    switch(b) { \
        SWITCH_START(a,s,h,t,TRUE)\
        SWITCH_START(a,s,h,t,FALSE)\
    }

#define KERNEL_SWITCH(a,s,h,t,b) \
    case a:\
        SWITCH_SECONDBEST(a,s,h,t,b)\
    break;

#include <vector>
//#include "adept_alignments.hpp"
#include <chrono>

#define NUM_OF_AA 21
#define ENCOD_MAT_SIZE 91
#define SCORE_MAT_SIZE 576

#define NSTREAMS 2

#define NOW std::chrono::high_resolution_clock::now()

struct alignment_results {
	short* ref_begin;
	short* query_begin;
	short* ref_end;
	short* query_end;
	short* top_scores;
};

#ifndef ALIGNMENTS_HPP
#define ALIGNMENTS_HPP

class gpu_alignments{
	public:
	short* ref_start_gpu;
	short* ref_end_gpu;
	short* query_start_gpu;
	short* query_end_gpu;
	short* scores_gpu;
	unsigned* offset_ref_gpu;
	unsigned* offset_query_gpu;

	gpu_alignments(int max_alignments);
	~gpu_alignments();
};

#endif

/* // Deprecated
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *query_batch, const uint32_t *query_batch_offsets, const uint32_t *query_batch_lens, const uint8_t *target_batch, const uint32_t *target_batch_offsets, const uint32_t *target_batch_lens,   const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_query_batch_start, int32_t *host_target_batch_start, int32_t *host_query_batch_end, int32_t *host_target_batch_end,  algo_type algo, comp_start start, int32_t k_band);
*/

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params);

inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band, int32_t zdrop);

int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage);

//NEW CODE

unsigned getMaxLength (std::vector<std::string> v);

void initialize_alignments(alignment_results *alignments, int max_alignments);

void asynch_mem_copies_htd(gpu_alignments* gpu_data, unsigned* offsetA_h, unsigned* offsetB_h, char* strA, char* strA_d, char* strB, char* strB_d, unsigned half_length_A, unsigned half_length_B, unsigned totalLengthA, unsigned totalLengthB, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda);

int get_new_min_length(short* alAend, short* alBend, int blocksLaunched);

void asynch_mem_copies_dth_mid(gpu_alignments* gpu_data, short* alAend, short* alBend, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda);

void asynch_mem_copies_dth(gpu_alignments* gpu_data, short* alAbeg, short* alBbeg, short* top_scores_cpu, int sequences_per_stream, int sequences_stream_leftover, cudaStream_t* streams_cuda);

void kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap);

void write_sam_and_stats(std::vector<std::string> refs, std::vector<std::string> quers, alignment_results *results);

#endif
