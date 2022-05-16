# xClean 3-pass denoiser for VapourSynth and Avisynth

Supported formats: YUV, RGB, GRAY  
Requires: rgsf, rgvs, fmtc, mv, mvsf, mvsfunc, knlm, bm3d, bm3dcuda_rtc, bm3dcpu, neo_tmedian, neo_f3kdb, akarin, nnedi3_resample, nnedi3cl

xClean runs MVTools -> BM3D -> KNLMeans in that order, passing the output of each pass as the ref of the next denoiser.

The objective is to remove noise while preserving as much details as possible. Removing noise is easy -- just blur out everything.  
The hard work is in preserving the details in a way that feels natural.

Designed for raw camera footage to remove noise in dark areas while preserving the fine details. It works for most types of content.

Performance-wise, BM3D pass is the heaviest and helps recover fine details, but this script runs 1 pass of BM3D whereas stand-alone BM3D runs twice.


### Short Doc (TL;DR)
Default settings provide the best quality in most cases. Simply use  
`xClean(sharp=..., outbits=...)`

If only darker areas contain noise, set strength=-50  
For better performance, set m1=0 or m2=0, or set m1=.5 and m2=3.6 (downscale)  
BM3D performance can be greatly improved by setting radius=0, block_step=7, bm_range=7, ps_range=5

For 720p WebCam, optimal settings are: sharp=9.5, m1=.65, h=2.8  
For 288p anime, optimal settings are: sharp=9.5, m1=.7, rn=0, optional depth=1  
For 4-5K GoPro (with in-camera sharpening at Low), optimal settings are: sharp=7.7, m1=.5, m2=2.7, optional strength=-50 (or m1=.6, m2=2.8 if your computer can handle it)


### Description
KNLMeans does a good job at denoising but can soften the image, lose details and give an artificial plastic look. I found that on any given source
(tested 5K GoPro footage and noisy WebCam), denoising with less than h=1.4 looks too noisy, and anything above it blurs out the details. 
KNLMeans also keeps a lot of data from the clip passed as rclip, so doing a good prefilter highly impacts the output.

Similarly, BM3D performs best with sigma=9. A lower value doesn't remove enough noise, and a higher value only makes the edges sharper.

xClean is essentially KNLMeans with advanced pre-filtering and with post-processing to renoise & sharpen to make the image look more natural.

One strange aspect of xClean is that denoising is automatic and there's very little room to configure denoising strength other than reducing the overall effect.
It runs with BM3D sigma=9 and KNL h=1.4, and generally you shouldn't change that. One setting that can allow increasing denoising (and performance)
is downscaling MVTools and BM3D passes. You can also set h=2.8 if the output remains too noisy. h = 1.4 or 2.8 are generally the best values.

According to my tests, water & cliff 5K video with little noise preserves the details very well while removing subtle grain, and with same settings,
very noisy 720p WebCam footage has HUGE noise reduction while preserving a surprising amount of natural details.

The default settings are very tolerant to various types of clips.

All processing is done in YUV444 format. When conv=True, processing is done in YCgCoR, and in OPP colorspace for BM3D.


### Denoising Methods Overview

To provide the best output, processing is done in 3 passes, passing the output of one pass as the ref clip of the 2nd pass. Each denoiser has its strengths and weaknesses.

Pass 1: MVTools (m1)  
Strength: Removes a lot of noise, good at removing temporal noise.  
Weakness: Can remove too much, especially with delicate textures like water.  
Ref: Impacts vectors analysis but low impact on outcome

Pass 2: BM3D (m2)  
Strength: Good at preserving fine details!  
Weakness: Doesn't remove much grain.  
Ref: Moderate impact on outcome. A blurry ref will remove more grain while BM3D puts back a lot of details.

Pass 3: KNLMeansCL (m3)  
Strength: Best general-purpose denoiser  
Weakness: Can blur out details and give an artificial plastic effect  
Ref: High impact the outcome. All prefilters benefit from running KNLMeans over it.


### Denoising Pass Configuration  (m1=.6, m2=2, m3=2)

Each pass (method) can be configured with m1 (MVTools), m2 (BM3D) and m3 (KNLMeansCL) parameters to run at desired bitdepth.
This means you can fine-tune for quality vs performance.

0 = Disabled, 1 = 8-bit, 2 = 16-bit, 3 = 32-bit

Note: BM3D always processes in 32-bit, KNLMeansCL always processes in 16-bit+, and post-processing always processes at least in 16-bit, so certain
values such as m2=1, m3=1 will behave the same as m2=2, m3=2. Setting m2=2 instead of 3 will only affect BM3D post-processing (YUV444P16 instead of YUV444PS)

MVTools (m1) and BM3D (m2) passes can also be downscaled for performance gain, and it can even improve quality! Values between .5 and .8 generally work best.

Optional resize factor is set after the dot:  
m1 = .6 or 1.6 processes MVTools in 8-bit at 60% of the size. m2 = 3.6 processes BM3D in 16-bit at 60% of the size.  
You may want to downscale MVTools (m1) because of high CPU usage and low impact on outcome.  
You may want to downscale BM3D (m2) because of high memory usage. If you run out of memory, lower the size until you get no hard-drive paging.  
Note: Setting radius=0 greatly reduces BM3D memory usage!


### Renoise and Sharpen  (rn=14, sharp=9.5)

The idea comes from mClean by Burfadel (https://forum.doom9.org/showthread.php?t=174804) and the algorithm was changed by someone else while porting 
to VapourSynth, producing completely different results -- original Avisynth version blurs a lot more, VapourSynth version keeps a lot more details.

It may sound counter-productive at first, but the idea is to combat the flat or plastic effect of denoising by re-introducing part of the removed noise.
The noise is processed and stabilized before re-inserting so that it's less distracting.
Renoise also helps reduce large-radius grain; but should be disabled for anime (rn=0).

Using the same analysis data, it's also sharpening to compensate for denoising blur.
Sharpening must be between 0 and 20. Actual sharpening calculation is scaled based on resolution.


### Strength / Dynamic Denoiser Strength  (strength=20)

A value of 20 will denoise normally.  
A value between 1 and 19 will reduce the denoising effect by partially merging back with the original clip.  
A value between 0 and -200 will activate Dynamic Denoiser Strength, useful when bright colors require little or no denoising and dark colors contain more noise.  
It applies a gradual mask based on luma. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction 
and the 50 whitest values are merged at a minimal strength of 50/255 = 20%.

### Radius  (radius=0)

BM3D radius. Low impact on individual frames.  
Pros: Helps stabilize temporal grain. Can significantly improve video compressability.  
Cons: High impact on performance and memory usage! May require downscaling BM3D for HD content with m2 between 3.6 and 3.8  
For moving water, the temporal stabilization may be undesirable.

### Depth  (depth=0)

This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

### Deband  (deband=False)

This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. Default settings are suitable for most cases without having a large effect on compressibility.

### Output  (outbits, dmode=0)

Specifies the output bitdepth. If not specified it will be converted back to the bitdepth of the source clip using dithering method specified by dmode.
You can set dmode=3 if you won't be doing any further processing for high-quality ditherig.

### Chroma upsampling/downsamping  (chroma=nnedi3, downchroma=True)

Chroma upsampling options:  
none = don't touch chroma  
bicubic = bicubic(0, .5) upsampling  
nnedi3 = NNEDI3 upsampling  
reconstructor = feisty2's ChromaReconstructor_faster v3.0 HBD mod

downchroma: whether to downscale back to match source clip. Default is False for reconstructor and True for other methods.

### Anime

For anime, set rn=0. Optionally, you can set depth to 1 or 2 to thicken the lines.

### Advanced Settings

gpuid = 0: The GPU id to use for KNLMeans and BM3D, or -1 to use CPU.  
gpucuda = 0: The GPU id to use for BM3D, or -1 to use CPU.  
h = 1.4: KNLMeans strength, can increase slightly if the output is still too noisy. 1.4 or 2.8 generally work best.  
block_step = 4, bm_range = 16, ps_range = 8: BM3D parameters for performance vs quality. No impact on CPU and memory. Adjust based on GPU capability.  
Fast settings are block_step = 5, bm_range = 7, ps_range = 5

Normally you shouldn't have to touch these  
rgmode = 18: RemoveGrain mode used during post-processing. Setting this to 0 disables post-processing, useful to compare raw denoising.  
thsad = 400: Threshold used for MVTools analysis.  
d = 2: KNLMeans temporal radius. Setting 3 can either slightly improve quality or give a slight plastic effect.  
a = 2: KNLMeans spacial radius.  
sigma = 9: BM3D strength.  
bm3d_fast = False. BM3D fast.  
conv = True. Whether to convert to OPP format for BM3D and YCgCoR for everything else. If false, it will process in standard YUV444.
