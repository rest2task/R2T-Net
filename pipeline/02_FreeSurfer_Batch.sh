#!/usr/bin/env bash
set -euo pipefail

export StudyDir=/localnvme/ADNI_niix

source "${HCPPIPEDIR}/Examples/Scripts/SetUpHCPPipeline.sh"

run_subj() {
  subj="$1"
  subj_path="${StudyDir}/${subj}"
  T1wImage=${subj_path}/T1w/T1w_acpc_dc_restore.nii.gz
  T1wBrain=${subj_path}/T1w/T1w_acpc_dc_restore_brain.nii.gz
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] START   ${subj}"
  "${HCPPIPEDIR}/FreeSurfer/FreeSurferPipeline.sh" \
    --session=${subj} \
    --session-dir="${subj_path}/T1w" \
    --t1w-image=${T1wImage} \
    --t1w-brain=${T1wBrain} \
    --t2w-image=NONE \
    --processing-mode=LegacyStyleData \
    > "$subj_path/02_FreeSurferPipeline.log" 2>&1
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] DONE    ${subj}"
}

export -f run_subj

cd "${StudyDir}"
ls -d */ | sed 's:/$::' \
  | parallel --jobs 50 --halt soon,fail=1 run_subj {}