#!/usr/bin/env bash
set -euo pipefail

#export StudyDir=/localnvme/ADNI_final
#export StudyDir=/home/y2jiang/ADNI_niix2
#export StudyDir=/home/y2jiang/UBDA/3T_HCP_Proc/ADNI_final
export StudyDir=/home/lab/UBDA/3T_HCP_Proc/ADNI_final

source "${HCPPIPEDIR}/Examples/Scripts/SetUpHCPPipeline.sh"

# ---------- ARG PARSING ----------
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) dry_run=true; shift ;;
    -*)           echo "Usage: $0 [-n|--dry-run] [part total]" ; exit 1 ;;
    *)            break ;;
  esac
done

if   (( $# == 0 )); then part_idx=0 ; total_parts=1        # run all
elif (( $# == 2 )); then
     part_idx=$(( $1 - 1 )) ; total_parts=$2
     (( part_idx >= 0 && part_idx < total_parts )) \
       || { echo "Invalid part/total."; exit 1; }
else
     echo "Usage: $0 [-n|--dry-run] [part total]" ; exit 1
fi

# ---------- SUBJECT SELECTION ----------
mapfile -t all_subj < <(cd "$StudyDir" && ls -d */ | sed 's:/$::' | sort)
selected=() skipped=()
for i in "${!all_subj[@]}"; do
  if (( i % total_parts == part_idx )); then
    selected+=( "${all_subj[i]}" )
  else
    skipped+=(  "${all_subj[i]}" )
  fi
done

echo "----- WILL RUN (${#selected[@]}) : ${selected[*]}"
echo "----- SKIPPED  (${#skipped[@]}) : ${skipped[*]}"

$dry_run && { echo "Dry-run: nothing executed."; exit 0; }

# ---------- PIPELINE ----------
run_subj() {
  subj="$1"
  echo "[`date +'%F %T'`] START  $subj"
  "${HCPPIPEDIR}/fMRISurface/SurfaceProcessingPipeline.sh" \
      --path="$StudyDir" \
      --subject="$subj" \
      --fmriname=rsfMRI \
      --lowresmesh=32 --fmrires=2 --smoothingFWHM=2 \
      --grayordinatesres=2 --regname=MSMSulc \
      > "$StudyDir/$subj/06_SurfaceProcessingPipeline.log" 2>&1
  echo "[`date +'%F %T'`] DONE   $subj"
}

export -f run_subj

printf "%s\n" "${selected[@]}" \
  | parallel --jobs 4 --verbose --halt soon,fail=1 run_subj {}
