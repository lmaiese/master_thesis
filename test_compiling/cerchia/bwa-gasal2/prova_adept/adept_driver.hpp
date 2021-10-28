#ifndef DRIVER_HPP
#define DRIVER_HPP

#include "adept_kernel.hpp"
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>
#include "adept_alignments.hpp"

#define NSTREAMS 2

#define NOW std::chrono::high_resolution_clock::now()



namespace gpu_bsw_driver{

// for storing the alignment results
struct alignment_results{
  short* ref_begin;
  short* query_begin;
  short* ref_end;
  short* query_end;
  short* top_scores;
};

void kernel_driver_dna(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scores[4]);

void kernel_driver_aa(std::vector<std::string> reads, std::vector<std::string> contigs, gpu_bsw_driver::alignment_results *alignments, short scoring_matrix[], short openGap, short extendGap);

void verificationTest(std::string rstFile, short* g_alAbeg, short* g_alBbeg, short* g_alAend,
                 short* g_alBend);
}
#endif
