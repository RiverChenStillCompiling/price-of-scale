#!/usr/bin/env bash
# Reproduce every number and figure in the paper.
#
# Inputs (see README):
#   data/raw/longmemeval_s_cleaned.json, data/raw/longmemeval_m_cleaned.json, data/raw/locomo10.json
#   SMALL, LARGE  paths to bge-small-en-v1.5 and bge-large-en-v1.5
#   Q7, Q14       paths to Qwen2.5-7B-Instruct-AWQ and Qwen2.5-14B-Instruct-AWQ
#   RERANKER      path to bge-reranker-base
#   PY, PY_LLM    python of the main environment and of the vLLM environment (default: python)
# Outputs go to out/.
set -euo pipefail
cd "$(dirname "$0")/experiments"
R=../data/raw
O=../out
PY=${PY:-python}
PY_LLM=${PY_LLM:-$PY}
NS_M=8,16,32,64,128,256,512,1024,2048
NS_S=8,16,32,64,128,256
export BG_MODE=full   # background drawn from the user's whole history outside the store
mkdir -p "$O"

replay() {  # name ns reps tag (SWAP_SAME and BG_MODE from the environment)
  local name=$1 ns=$2 reps=$3 tag=$4
  $PY trials.py "$O/scores_$name.npz" "$O/trials_$tag.csv" "$ns" "$reps"
  mkdir -p "$O/$tag" && $PY analyse.py "$O/trials_$tag.csv" "$O/$tag"
}
run() {  # name data_file model ns reps [cue_file]: embed, then replay with both backgrounds
  local name=$1 data=$2 model=$3 ns=$4 reps=$5 cues=${6:-}
  $PY embed.py "$data" "$O/scores_$name.npz" "$model" $cues
  BG_MODE=full replay $name $ns $reps ${name}_full
  BG_MODE=pool replay $name $ns $reps ${name}_pool
}

# 1. Gates on the three data sets with both embedding models (Appendix C compares the two backgrounds)
run s       $R/longmemeval_s_cleaned.json "$SMALL" $NS_S 20
run m       $R/longmemeval_m_cleaned.json "$SMALL" $NS_M 10
run m_large $R/longmemeval_m_cleaned.json "$LARGE" $NS_M 10
run locomo       $R/locomo10.json "$SMALL" $NS_S 20
run locomo_large $R/locomo10.json "$LARGE" $NS_S 20

# 2. LoCoMo with reference cues from the same conversation
for name in locomo locomo_large; do
  for bg in full pool; do SWAP_SAME=1 BG_MODE=$bg replay $name $NS_S 20 ${name}_same_$bg; done
done

# 3. Theory: finite-pool consistency check (Figure 3), out-of-sample prediction and simulation (Appendix E)
for name in m m_large; do
  $PY verify_theory2.py "$O/scores_$name.npz" "$O/trials_${name}_full.csv" "$O/${name}_full/theory2.json"
  $PY heldout_theory.py "$O/scores_$name.npz" "$O/trials_${name}_full.csv" "$O/heldout_$name.json"
done
$PY sim_theorem2.py "$O/sim_theorem2.json"

# 4. LLM judges
for q in 7 14; do
  QM=$([ $q = 7 ] && echo "$Q7" || echo "$Q14")
  $PY_LLM llm_judge.py $R/longmemeval_m_cleaned.json "$O/scores_m.npz" "$QM" "$O/llm_q${q}_m.csv" 8,64,512,2048 3
  $PY_LLM llm_judge.py $R/locomo10.json "$O/scores_locomo.npz" "$QM" "$O/llm_q${q}_locomo.csv" 8,64,256 2
done

# 5. Implicit cues (the released data/implicit_cues_longmemeval_m.json was produced by this step)
CUES=../data/implicit_cues_longmemeval_m.json
[ -f "$CUES" ] || $PY_LLM rewrite_cues.py $R/longmemeval_m_cleaned.json "$Q14" "$CUES"
run m_implicit       $R/longmemeval_m_cleaned.json "$SMALL" $NS_M 10 "$CUES"
run m_implicit_large $R/longmemeval_m_cleaned.json "$LARGE" $NS_M 10 "$CUES"
$PY_LLM llm_judge.py $R/longmemeval_m_cleaned.json "$O/scores_m_implicit.npz" "$Q14" "$O/llm_q14_m_implicit.csv" 8,64,512,2048 3 "$CUES"

# 6. Scorer components at matched false alarms (Table 1), intervals and growth of the null maximum (Appendix F)
for name in m m_large m_implicit locomo; do $PY ablation_norm.py "$O/trials_${name}_full.csv" "$O/matched_$name"; done
for name in m m_large m_implicit m_implicit_large s locomo locomo_large locomo_same locomo_large_same; do
  $PY gates_ci.py "$O/trials_${name}_full.csv" "$O/ci_$name"
done

# 7. Cue-swap gate: number of reference cues, nominal level and latency (Appendix D)
for name in m m_implicit; do
  $PY swap_ablation.py "$O/scores_$name.npz" "$O/ablation_$name.csv" 8,64,512,2048 5
done
$PY ablation_table.py "$O/ablation_m.csv" "$O/ablation_m_implicit.csv"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY swap_latency.py > "$O/latency.txt"

# 8. Figures and Table 2
$PY make_figs.py "$O" "$O/figs"

# 9. Chronological replay (Table 3)
$PY embed_items.py $R/longmemeval_m_cleaned.json "$SMALL" "$O/items_m" "$CUES"
mkdir -p "$O/chronological"
for s in 0 1 2 3 4 5 6 7; do
  $PY temporal.py $R/longmemeval_m_cleaned.json "$O/items_m" "$O/scores_m.npz" "$O/trials_m_full.csv" "$O/chronological/t_$s.csv" $s 8
done
$PY temporal_table.py "$O/trials_m_full.csv" "$O/chronological/summary" "$O"/chronological/t_?.csv

# 10. Several relevant memories (Appendix G)
for mode in all dup3 session; do
  BG_MODE=full POS_MODE=$mode KINDS=pos $PY trials.py "$O/scores_m.npz" "$O/trials_m_pos_$mode.csv" $NS_M 10
done
BG_MODE=full POS_MODE=session HIT_REL=1 KINDS=pos $PY trials.py "$O/scores_m.npz" "$O/trials_m_pos_session_rel.csv" $NS_M 10
$PY sensitivity.py "$O/trials_m_full.csv" "$O/sensitivity.csv" all="$O/trials_m_pos_all.csv" dup3="$O/trials_m_pos_dup3.csv" \
  session="$O/trials_m_pos_session.csv" session_rel="$O/trials_m_pos_session_rel.csv"

# 11. TriggerBench (Appendix H): benchmark/prospective_memory/data of KristenZHANG/TriggerBench-Official
#     (commit 9e2185c) in data/raw/triggerbench
$PY triggerbench.py $R/triggerbench "$SMALL" "$O/tb_small"
$PY triggerbench_table.py "$O/tb_small_trials.csv" "$O/tb_small_summary.csv"

# 12. Cross-encoder gate (Section 5.9); RERANKER is the path to BAAI/bge-reranker-base
for name in m m_implicit; do
  c=$([ $name = m_implicit ] && echo "$CUES" || true)
  $PY rerank.py $R/longmemeval_m_cleaned.json "$O/scores_$name.npz" "$RERANKER" "$O/rr_$name.csv" 8,64,512,2048 5 $c
  $PY rerank_table.py "$O/rr_$name.csv" "$O/rr_${name}_summary.csv"
done
