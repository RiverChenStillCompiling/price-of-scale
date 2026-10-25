#!/usr/bin/env bash
# Reproduce every number and figure in the paper.
#
# Inputs (see README):
#   data/raw/longmemeval_s_cleaned.json, data/raw/longmemeval_m_cleaned.json, data/raw/locomo10.json
#   SMALL, LARGE  paths to bge-small-en-v1.5 and bge-large-en-v1.5
#   Q7, Q14       paths to Qwen2.5-7B-Instruct-AWQ and Qwen2.5-14B-Instruct-AWQ
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
export BG_POOL=1   # background sample drawn from the null pool only
mkdir -p "$O"

run() {  # name data_file model ns reps [cue_file]
  local name=$1 data=$2 model=$3 ns=$4 reps=$5 cues=${6:-}
  $PY embed.py "$data" "$O/scores_$name.npz" "$model" $cues
  $PY trials.py "$O/scores_$name.npz" "$O/trials_$name.csv" "$ns" "$reps"
  mkdir -p "$O/$name" && $PY analyse.py "$O/trials_$name.csv" "$O/$name"
}

# 1. Gates on the three data sets with both embedding models
run s       $R/longmemeval_s_cleaned.json "$SMALL" $NS_S 20
run s_large $R/longmemeval_s_cleaned.json "$LARGE" $NS_S 20
run m       $R/longmemeval_m_cleaned.json "$SMALL" $NS_M 10
run m_large $R/longmemeval_m_cleaned.json "$LARGE" $NS_M 10
run locomo       $R/locomo10.json "$SMALL" $NS_S 20
run locomo_large $R/locomo10.json "$LARGE" $NS_S 20

# 2. LoCoMo with reference cues from the same conversation
for name in locomo locomo_large; do
  SWAP_SAME=1 $PY trials.py "$O/scores_$name.npz" "$O/trials_${name}_same.csv" $NS_S 20
  mkdir -p "$O/${name}_same" && $PY analyse.py "$O/trials_${name}_same.csv" "$O/${name}_same"
done

# 3. Theory checks (Figure 3)
for name in m m_large; do
  $PY verify_theory.py  "$O/scores_$name.npz" "$O/$name/gates.csv"  "$O/$name/theory.json"
  $PY verify_theory2.py "$O/scores_$name.npz" "$O/trials_$name.csv" "$O/$name/theory2.json"
done

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

# 6. Cue-swap gate: number of reference cues and nominal level (Appendix C)
for name in m m_implicit; do
  $PY swap_ablation.py "$O/scores_$name.npz" "$O/ablation_$name.csv" 8,64,512,2048 5
done
$PY ablation_table.py "$O/ablation_m.csv" "$O/ablation_m_implicit.csv"

# 7. Figures and Table 1
$PY make_figs.py "$O" "$O/figs"
