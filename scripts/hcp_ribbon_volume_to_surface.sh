#!/usr/bin/env bash
set -euo pipefail

#export StudyDir=/localnvme/ADNI_niix
#export StudyDir=/home/y2jiang/ADNI_niix2
export StudyDir=/localnvme/ADNI_final

source "${HCPPIPEDIR}/Examples/Scripts/SetUpHCPPipeline.sh"

run_subj() {
  subj="$1"
  subj_path="${StudyDir}/${subj}"
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] START   ${subj}"
  "${HCPPIPEDIR}/fMRISurface/RibbonVolumeToSurfaceMapping.sh" \
    --path="${StudyDir}" \
    --subject="${subj}" \
    --fmriname="rsfMRI" \
    --lowresmesh="32" \
    --fmrires="2" \
    --smoothingFWHM="2" \
    --grayordinatesres="2" \
    --regname="MSMSulc" \
    > "$subj_path/05_RibbonVolumeToSurfaceMapping.log" 2>&1
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] DONE    ${subj}"
}

export -f run_subj

cd "${StudyDir}"
ls -d */ | sed 's:/$::' \
  | parallel --jobs 20 --halt soon,fail=1 run_subj {}