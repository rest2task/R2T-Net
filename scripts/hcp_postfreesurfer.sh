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
  "${HCPPIPEDIR}/PostFreeSurfer/PostFreeSurferPipeline.sh" \
    --study-folder=${StudyDir} \
    --subject=${subj} \
    --surfatlasdir=${HCPPIPEDIR_Templates}/standard_mesh_atlases \
    --grayordinatesdir=${HCPPIPEDIR_Templates}/91282_Greyordinates \
    --grayordinatesres=2 \
    --hiresmesh=164 \
    --lowresmesh=32 \
    --subcortgraylabels=${HCPPIPEDIR_Config}/FreeSurferSubcorticalLabelTableLut.txt \
    --freesurferlabels=${HCPPIPEDIR_Config}/FreeSurferAllLut.txt \
    --refmyelinmaps=${HCPPIPEDIR_Templates}/standard_mesh_atlases/Conte69.MyelinMap_BC.164k_fs_LR.dscalar.nii \
    --regname=MSMSulc \
    --use-ind-mean=YES \
    --processing-mode=LegacyStyleData \
    > "$subj_path/03_PostFreeSurferPipeline.log" 2>&1
  echo "[`date +'%Y-%m-%d %H:%M:%S'`] DONE    ${subj}"
}

export -f run_subj

cd "${StudyDir}"
ls -d */ | sed 's:/$::' \
  | parallel --jobs 50 --halt soon,fail=1 run_subj {}