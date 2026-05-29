#!/usr/bin/env bash
set -euo pipefail

#export StudyDir=/localnvme/ADNI_niix
export StudyDir=/home/y2jiang/ADNI_niix2

source "${HCPPIPEDIR}/Examples/Scripts/SetUpHCPPipeline.sh"

run_subj() {
  subj="$1"
  subj_path="${StudyDir}/${subj}"
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] START   ${subj}"
  "${HCPPIPEDIR}/PreFreeSurfer/PreFreeSurferPipeline.sh" \
    --path="${StudyDir}" \
    --session="${subj}" \
    --t1="${subj_path}/T1w.nii.gz" \
    --t2=NONE \
    --fmapmag="${subj_path}/FieldMap_mag.nii.gz" \
    --fmapphase="${subj_path}/FieldMap_ph.nii.gz" \
    --avgrdcmethod=SiemensFieldMap \
    --echodiff=2.46 \
    --SEPhaseNeg=NONE \
    --SEPhasePos=NONE \
    --seechospacing=0.00058 \
    --seunwarpdir=NONE \
    --topupconfig=NONE \
    --t1template="${HCPPIPEDIR_Templates}/MNI152_T1_0.7mm.nii.gz" \
    --t1templatebrain="${HCPPIPEDIR_Templates}/MNI152_T1_0.7mm_brain.nii.gz" \
    --t1template2mm="${HCPPIPEDIR_Templates}/MNI152_T1_2mm.nii.gz" \
    --t2template="${HCPPIPEDIR_Templates}/MNI152_T2_0.7mm.nii.gz" \
    --t2templatebrain="${HCPPIPEDIR_Templates}/MNI152_T2_0.7mm_brain.nii.gz" \
    --t2template2mm="${HCPPIPEDIR_Templates}/MNI152_T2_2mm.nii.gz" \
    --templatemask="${HCPPIPEDIR_Templates}/MNI152_T1_0.7mm_brain_mask.nii.gz" \
    --template2mmmask="${HCPPIPEDIR_Templates}/MNI152_T1_2mm_brain_mask_dil.nii.gz" \
    --brainsize=150 \
    --fnirtconfig="${HCPPIPEDIR_Config}/T1_2_MNI152_2mm.cnf" \
    --t1samplespacing=0.00417 \
    --t2samplespacing=NONE \
    --unwarpdir=z \
    --gdcoeffs=NONE \
    --processing-mode=LegacyStyleData \
    > "$subj_path/01_PreFreeSurferPipeline.log" 2>&1
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] DONE    ${subj}"
}

export -f run_subj

cd "${StudyDir}"
ls -d */ | sed 's:/$::' \
  | parallel --jobs 50 --halt soon,fail=1 run_subj {}