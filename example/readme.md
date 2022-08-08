# Example
Here is a workflow of using RefHiC to detect loop and TAD boundary from Hi-C contact maps.

In this example, we  show how to detect loops on chromosome 17 from a GM12878 Hi-C contact map (500M valid read pairs). 

We will use the provided <i>.bcool</i> file (gm12878_500M.bcool) as input. It is an modified <i>.cool</i> format that only contains contacts within 3Mb distance to reduce file size. You can use a <i>.cool</i> file as well. 

We assume you already installed and configured (`i.e. refhic config init`) RefHiC. 
##  Loop annotations
The following command will produce three files: 
   1) loop scores for candidiate loci (gm12878_500M_chr17_loopCandidates.bedpe); 
   2) loops (gm12878_500M_chr17_loop.bedpe); 
   3) pileup around called loops (gm12878_500M_chr17_loop.bedpe.png)
<pre>
# calculate loop scores
refhic loop pred --chrom chr17 gm12878_500M.bcool gm12878_500M_chr17_loopCandidates.bedpe
# select loops
refhic loop pool gm12878_500M_chr17_loopCandidates.bedpe gm12878_500M_chr17_loop.bedpe
# optional, santity check 
refhic util pileup  --p2ll True gm12878_500M_chr17_loop.bedpe gm12878_500M.bcool
</pre>

##  TAD annotations
The following command will produce three files: 
   1. Boundary scores for candidiate loci (gm12878_500M_chr17_BoundaryScores.bed); 
   2. left boundaries (gm12878_500M_chr17_leftBoundary.bed); 
   3. right boundaries (gm12878_500M_chr17_rightBoundary.bed); 
   4. pileup around left boundaries (gm12878_500M_chr17_leftBoundary.bed_pileup.png)
   5. pileup around right boundaries (gm12878_500M_chr17_rightBoundary.bed_pileup.png)
<pre>
# calculate TAD scores
refhic tad pred --chrom chr17 gm12878_500M.bcool gm12878_500M_chr17_BoundaryScores.bed
# extract boundary from boundaryScore file
cat gm12878_500M_chr17_BoundaryScores.bed | awk '{if($5==1)print}' > gm12878_500M_chr17_leftBoundary.bed
cat gm12878_500M_chr17_BoundaryScores.bed | awk '{if($NF==1)print}' > gm12878_500M_chr17_rightBoundary.bed
# optional, santity check
refhic util pileup gm12878_500M_chr17_rightBoundary.bed gm12878_500M.bcool
refhic util pileup gm12878_500M_chr17_leftBoundary.bed gm12878_500M.bcool
</pre>
