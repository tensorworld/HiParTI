/*
    This file is part of HiParTI!.

    Reader for the TensorSuite tensor format (https://tensorworld.github.io/TensorSuite/).

    A TensorSuite text tensor differs from HiParTI's native .tns in three ways:

      native .tns              TensorSuite .tns
      -----------              ----------------
      <nmodes>                 %%TensorSuite-TNS          <- banner
      <dim1> .. <dimN>         % version: 0.1             <- '%' comment lines
      i1 .. iN val             % name: <tensor-name>
                               <order> <dim1> .. <dimN> <nnz>   <- ONE size line, includes nnz
                               i1 .. iN [val]             <- the value column is OPTIONAL

    Two further properties are carried in the sidecar "<stem>_metadata.json":
      * "index_base"      : 0 or 1  (both occur in the collection)
      * "values_provided" : false for many tensors, which store only the pattern
      * "sparsity_type"   : "element" or "block"
    The sidecar is read when present; otherwise both are inferred from the data.

    When no values are stored, they are synthesised according to `fill`.
*/

#include <HiParTI.h>
#include "sptensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define PTI_TS_BANNER "%%TensorSuite"

/* splitmix64, identical to ptiRandomizeMatrix, so synthesised values are reproducible */
static ptiValue pti_TsNextRandom(uint64_t * state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (ptiValue)((double)(z >> 11) / 9007199254740992.0);
}

/**
 * Peek at a file and report whether it carries the TensorSuite banner.
 * The stream position is restored, so the caller can fall back to another reader.
 */
int pti_TensorSuiteSniff(FILE * fp)
{
    long const pos = ftell(fp);
    char head[32];
    size_t const n = fread(head, 1, sizeof(head) - 1, fp);
    head[n] = '\0';
    if(fseek(fp, pos, SEEK_SET) != 0) return 0;
    return strncmp(head, PTI_TS_BANNER, strlen(PTI_TS_BANNER)) == 0;
}

/* Pull one scalar field out of the sidecar JSON. Returns 0 when found. */
static int pti_TsMetaField(char const * fname, char const * key, char * out, size_t outlen)
{
    char path[4096];
    char const * dot = strrchr(fname, '.');
    size_t const stem = dot ? (size_t)(dot - fname) : strlen(fname);
    if(stem + 32 > sizeof(path)) return -1;
    memcpy(path, fname, stem);
    snprintf(path + stem, sizeof(path) - stem, "_metadata.json");

    FILE * mf = fopen(path, "r");
    if(!mf) return -1;

    int found = -1;
    char line[1024];
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    while(fgets(line, sizeof(line), mf)) {
        char * p = strstr(line, needle);
        if(!p) continue;
        p = strchr(p + strlen(needle), ':');
        if(!p) continue;
        ++p;
        while(*p && (isspace((unsigned char)*p) || *p == '"')) ++p;
        size_t k = 0;
        while(*p && *p != ',' && *p != '"' && *p != '\n' && k + 1 < outlen) out[k++] = *p++;
        while(k > 0 && isspace((unsigned char)out[k-1])) --k;
        out[k] = '\0';
        found = 0;
        break;
    }
    fclose(mf);
    return found;
}

/* Next line that is neither blank nor a '%' comment. Returns 0 on success. */
static int pti_TsNextDataLine(FILE * fp, char * buf, size_t buflen)
{
    while(fgets(buf, (int)buflen, fp)) {
        char * p = buf;
        while(*p && isspace((unsigned char)*p)) ++p;
        if(*p == '\0' || *p == '%' || *p == '#') continue;
        return 0;
    }
    return -1;
}

static size_t pti_TsCountTokens(char const * s)
{
    size_t n = 0;
    while(*s) {
        while(*s && isspace((unsigned char)*s)) ++s;
        if(!*s) break;
        ++n;
        while(*s && !isspace((unsigned char)*s)) ++s;
    }
    return n;
}

/**
 * Load a TensorSuite-format sparse tensor into HiParTI's COO representation.
 *
 * @param tsr          the tensor to fill in
 * @param start_index  index base wanted in `tsr` (1 keeps MATLAB-style indices)
 * @param fname        path to the .tns file (the sidecar JSON is looked up beside it)
 * @param fill         PTI_TS_FILL_ONES or PTI_TS_FILL_RANDOM, used only when the
 *                     file stores no value column
 */
int ptiLoadSparseTensorTensorSuite(
    ptiSparseTensor *tsr,
    ptiIndex start_index,
    char const * const fname,
    ptiTensorSuiteFill fill)
{
    FILE * fp = fopen(fname, "r");
    if(!fp) {
        fprintf(stderr, "[TensorSuite] Error: cannot open %s\n", fname);
        return -1;
    }
    if(!pti_TensorSuiteSniff(fp)) {
        fprintf(stderr, "[TensorSuite] Error: %s does not start with \"%s\"\n", fname, PTI_TS_BANNER);
        fclose(fp);
        return -1;
    }

    char meta[128];
    /* Block-sparse files carry a block-grid line the COO reader cannot interpret. */
    if(pti_TsMetaField(fname, "sparsity_type", meta, sizeof(meta)) == 0 &&
       strcmp(meta, "block") == 0) {
        fprintf(stderr, "[TensorSuite] Error: %s is a block-sparse tensor; "
                        "only element-wise tensors can be read into COO.\n", fname);
        fclose(fp);
        return -1;
    }

    char line[8192];
    if(pti_TsNextDataLine(fp, line, sizeof(line)) != 0) {
        fprintf(stderr, "[TensorSuite] Error: %s has no size line\n", fname);
        fclose(fp);
        return -1;
    }

    /* size line: <order> <dim1> ... <dimN> <nnz> */
    size_t const ntok = pti_TsCountTokens(line);
    char * save = NULL;
    char * tok = strtok_r(line, " \t\n\r", &save);
    ptiIndex const nmodes = (ptiIndex) strtoul(tok, NULL, 10);
    if(nmodes == 0 || ntok != (size_t) nmodes + 2) {
        fprintf(stderr, "[TensorSuite] Error: malformed size line in %s "
                        "(order=%u but %zu fields; expected %u)\n",
                        fname, nmodes, ntok, nmodes + 2);
        fclose(fp);
        return -1;
    }

    ptiIndex * ndims = (ptiIndex *) malloc(nmodes * sizeof *ndims);
    if(!ndims) { fclose(fp); return -1; }
    for(ptiIndex m = 0; m < nmodes; ++m) {
        tok = strtok_r(NULL, " \t\n\r", &save);
        ndims[m] = (ptiIndex) strtoul(tok, NULL, 10);
    }
    tok = strtok_r(NULL, " \t\n\r", &save);
    ptiNnzIndex const nnz_declared = (ptiNnzIndex) strtoull(tok, NULL, 10);

    /* index base: metadata wins, else inferred from the data below */
    int file_base = -1;
    if(pti_TsMetaField(fname, "index_base", meta, sizeof(meta)) == 0) {
        file_base = atoi(meta);
    }

    /* Does the first data row carry a value column? */
    long const first_row = ftell(fp);
    if(pti_TsNextDataLine(fp, line, sizeof(line)) != 0) {
        fprintf(stderr, "[TensorSuite] Error: %s declares %" HIPARTI_PRI_NNZ_INDEX
                        " nonzeros but has no data rows\n", fname, nnz_declared);
        free(ndims); fclose(fp);
        return -1;
    }
    size_t const row_tok = pti_TsCountTokens(line);
    int has_values;
    if(row_tok == (size_t) nmodes)          has_values = 0;
    else if(row_tok == (size_t) nmodes + 1) has_values = 1;
    else {
        fprintf(stderr, "[TensorSuite] Error: %s data row has %zu fields, expected %u or %u\n",
                        fname, row_tok, nmodes, nmodes + 1);
        free(ndims); fclose(fp);
        return -1;
    }
    fseek(fp, first_row, SEEK_SET);

    int result = ptiNewSparseTensor(tsr, nmodes, ndims);
    free(ndims);
    if(result != 0) { fclose(fp); return result; }

    uint64_t rng = HIPARTI_RANDOM_SEED;
    ptiNnzIndex nread = 0;
    ptiIndex * idx = (ptiIndex *) malloc(nmodes * sizeof *idx);
    if(!idx) { fclose(fp); return -1; }
    int saw_zero_index = 0;

    while(pti_TsNextDataLine(fp, line, sizeof(line)) == 0) {
        save = NULL;
        tok = strtok_r(line, " \t\n\r", &save);
        int ok = 1;
        for(ptiIndex m = 0; m < nmodes; ++m) {
            if(!tok) { ok = 0; break; }
            unsigned long long const v = strtoull(tok, NULL, 10);
            if(v == 0) saw_zero_index = 1;
            idx[m] = (ptiIndex) v;
            tok = strtok_r(NULL, " \t\n\r", &save);
        }
        if(!ok) continue;

        ptiValue val;
        if(has_values) {
            val = tok ? (ptiValue) atof(tok) : (ptiValue) 0;
        } else {
            val = (fill == PTI_TS_FILL_RANDOM) ? pti_TsNextRandom(&rng) : (ptiValue) 1;
        }

        for(ptiIndex m = 0; m < nmodes; ++m) {
            result = ptiAppendIndexVector(&tsr->inds[m], idx[m]);
            if(result != 0) { free(idx); fclose(fp); return result; }
        }
        result = ptiAppendValueVector(&tsr->values, val);
        if(result != 0) { free(idx); fclose(fp); return result; }
        ++nread;
    }
    free(idx);
    fclose(fp);
    tsr->nnz = nread;

    if(file_base < 0) file_base = saw_zero_index ? 0 : 1;

    /* HiParTI stores indices 0-based internally (the native reader does
       `index - start_index`).  Unlike the native path we know the file's real
       base from the metadata, so use that rather than the caller's guess. */
    (void) start_index;
    if(file_base != 0) {
        for(ptiIndex m = 0; m < nmodes; ++m) {
            for(ptiNnzIndex z = 0; z < nread; ++z) {
                tsr->inds[m].data[z] -= (ptiIndex) file_base;
            }
        }
    }

    if(nnz_declared != nread) {
        fprintf(stderr, "[TensorSuite] Warning: %s declares %" HIPARTI_PRI_NNZ_INDEX
                        " nonzeros but %" HIPARTI_PRI_NNZ_INDEX " were read\n",
                        fname, nnz_declared, nread);
    }
    return 0;
}
