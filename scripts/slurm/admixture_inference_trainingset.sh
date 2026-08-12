OUTDIR=/work/magroup/skrieger/Syndri/data/trainsets/syndri_predictions/

for SPLIT_VALUE in train val; do
  sbatch --export=ALL,ST_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/chromium_GSM9509174_breast_integrated.h5ad,REFERENCE_DATA=/compute/oven-0-13/jwinchel/scRNA-seq/adata/chromium_GSM9509174_breast_integrated.h5ad,SPLIT_VALUE=${SPLIT_VALUE},PRESERVE_SPLIT_COLUMN=1,COORD_X=x_centroid,COORD_Y=y_centroid,APPLY_ADMIXTURE=1,MILESTONE=50,CHECKPOINT_DIR=/home/jwinchel/Syndri/Heimdall/admixture_prediction_results/chromium_breast_GS95019174_augmented,OUTPUT_DIR=${OUTDIR},PREPARED_NAME=chromium_breast_${SPLIT_VALUE}_prepared.h5ad,OUTPUT_NAME=chromium_breast_${SPLIT_VALUE}_with_admix_probs.h5ad \
    /home/jwinchel/Syndri/scripts/slurm/admixture_inference.sbatch
done