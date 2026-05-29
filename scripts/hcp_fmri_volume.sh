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
  "${HCPPIPEDIR}/fMRIVolume/GenericfMRIVolumeProcessingPipeline.sh" \
      --path="${StudyDir}" \
      --subject="${subj}" \
      --fmriname="rsfMRI" \
      --fmritcs="${subj_path}/rsfMRI.nii.gz" \
      --fmriscout="NONE" \
      --SEPhaseNeg="NONE" \
      --SEPhasePos="NONE" \
      --echospacing="0.00058" \
      --unwarpdir="y" \
      --fmrires="2" \
      --dcmethod="NONE" \
      --biascorrection="NONE" \
      --gdcoeffs="NONE" \
      --mctype="MCFLIRT" \
      --processing-mode=LegacyStyleData \
    > "$subj_path/04_GenericfMRIVolumeProcessingPipeline.log" 2>&1
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] DONE    ${subj}"
}

export -f run_subj

cd "${StudyDir}"
ls -d */ | sed 's:/$::' \
  | parallel --jobs 50 --halt soon,fail=1 run_subj {}