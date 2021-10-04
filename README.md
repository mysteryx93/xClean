# xClean 3-pass denoiser

beta 2 (2021-10-04) by Etienne Charland  
Supported formats: YUV or GRAY  
Requires: rgsv, rgvs, fmtc, mv, mvsf, tmedian

xClean runs MVTools -> BM3D -> KNLMeans in that order, passing the output of each pass as the ref of the next denoiser.

The objective is to remove noise while preserving as much details as possible. Removing noise is easy -- just blur out everything.
The hard work is in preserving the details in a way that feels natural.


### Short Doc (TL;DR)

Default settings provide the best quality in most cases. Simply use  
xClean(sharp=..., outbits=...)  
For top quality, you can add d=3.


### Long version

KNLMeans does a good job at denoising but can soften the image, lose details and give an artificial plastic look. I found that on any given source
(tested 5K GoPro footage and noisy WebCam), denoising with less than h=1.4 looks too noisy, and anything above it blurs out the details. 
I thus run it at 1.4 strength. KNLMeans also keeps a lot of data from the clip passed as rclip, so doing a good prefilter highly impacts the output.

Similarly, BM3D performs best with sigma=9. A lower value doesn't remove enough noise, and a higher value only makes the edges sharper.

xClean is essentially KNLMeans with advanced pre-filtering and with post-processing to renoise & sharpen to make the image look more natural.

One strange aspect of xClean is that denoising is automatic and there's very little room to configure denoising strength other than reducing it.
It runs with BM3D sigma=9 and KNL h=1.4, and generally you shouldn't change that. One setting that can allow increasing denoising (and performance)
is downscaling MVTools pass.

According to my tests, water & cliff 5K video with little noise preserves the details very well while removing subtle grain, and with same settings,
very noisy 720p WebCam footage has HUGE noise reduction while preserving a surprising amount of natural details.

The default settings are very tolerant to various types of clips.


+++ Denoising Methods Overview +++

To provide the best output, processing is done in 3 passes, passing the output of one pass as the ref clip of the 2nd pass. Each denoiser has its strengths and weaknesses.

Pass 1: MVTools  
Strength: Removes a lot of noise, good at removing temporal noise.  
Weakness: Can remove too much, especially with delicate textures like water.  
Ref: Impacts vectors analysis but low impact on outcome (running simple convolution matrix on ref)

Pass 2: BM3D  
Strength: Good at preserving fine details!  
Weakness: Doesn't remove much grain. Poor temporal stability.   
Ref: Moderate impact on outcome. A blurry ref will remove more grain while BM3D puts back a lot of details.  
radius=1 provides nearly no benefit at huge performance cost since MVTools already does temporal analysis

Pass 3: KNLMeansCL  
Strength: Best general-purpose denoiser  
Weakness: Can blur out details and give an artificial plastic effect  
Ref: Highly impacts the outcome. All prefilters benefit from running KNLMeans over it.  
By default it runs with d=2, a=2. You can set d=3 for slight quality improvement.

MVTools + BM3D  
Strength: Keeps a lot of details with good denoising of fine details as well as larger grain. Single frames can look great.  
Weakness: Poor noise temporal stability. The video doesn't look as good as single frames.

MVTools + KNLMeans  
Strength: KNLMeans with extra denoising. Works best in most circumstances.  
Weakness: Delicate textures like water or fog will suffer.

MVTools + BM3D + KNLMeans  
Strength: Like MvTools+KNLMeans but preserves details with delicate textures. Works best for any kind of content tested.  
Weakness: Performance and memory usage.


+++ Denoising Pass Configuration (m1, m2, m3) +++

Each pass (method) can be configured with m1 (MVTools), m2 (BM3D) and m3 (KNLMeansCL) parameters to run at desired bitdepth.
This means you can fine-tune for quality vs performance.

0 = Disabled, 1 = 8-bit, 2 = 16-bit, 3 = 16-bit YUV444, 4 = 32-bit YUV444

Note: BM3D always processes in 32-bit, KNLMeansCL always processes in 16-bit+, and post-processing always processes in 16-bit+, so certain
values such as m2=1, m3=1 will behave the same as m2=2, m3=2. Setting m2=3 will only affect BM3D post-processing (YUV444P16 instead of YUV420P16)

MVTools pass (m1) can also be downscaled for huge performance gain, and it even improves quality by bluring more noise before analysis.
Resizing by a factor of .6 provides the best quality in my tests, and .5 works best if you want that extra performance.

Optional resize factor is set after the dot:  
m1 = .6 or 1.6 processes in 8-bit at 60% of the size. m1 = 2.6 processes in 16-bit at 60% of the size.

Default configuration is m1=.6, m2=3, m3=3 which will provide the best quality in most cases.

For better performance, you can disable m1 for videos with delicate textures and low noise, or disable m2 for videos with no delicate textures.
You can also simply resize MVTools pass smaller (.5 or .4) which will produce a bit more blur. If m1=1 (no downsize), you can reduce sharp from 11 to 10.


+++ Renoise and Sharpen (sharp) +++

The idea comes from mClean by Burfadel (https://forum.doom9.org/showthread.php?t=174804) and the algorithm was changed by someone else while porting 
to VapourSynth, producing completely different results -- original Avisynth version blurs a lot more, VapourSynth version keeps a lot more details.

It may sound counter-productive at first, but the idea is to combat the flat or plastic effect of denoising by re-introducing part of the removed noise.
The noise is processed and stabilized before re-inserting so that it's less distracting.

Using the same analysis data, it's also sharpening to compensate for denoising blur.

Normal sharpening must be between 0 and 20. 21-24 provide 'overboost' sharpening, generally only suitable for high definition, high quality sources.
Actual sharpening calculation is scaled based on resolution.

Default: 11. Much less sharpening is required than mClean due to the way denoisers are chained.


+++ Strength / Dynamic Denoiser Strength (strength) +++

A value of 20 (default) will denoise normally. Set a value around -50 if you only dark areas contain noise.

A value between 1 and 19 will reduce the denoising effect by that factor by partially merging back with the original clip.

A value between 0 and -200 will activate Dynamic Denoiser Strength, useful when bright colors require little or no denoising and dark colors contain more noise.
It applies a gradual mask based on luma. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction 
and the 50 whitest values are merged at a minimal strength of 50/255 = 20%.


+++ depth +++

This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.


+++ deband +++

This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. The settings are not adjustable as the default settings are suitable for most cases without having a large effect
on compressibility. 0 = disabled, 1 = deband


+++ outbits, dmode +++

Specifies the output bitdepth. If not specified it will be converted back to the bitdepth of the source clip using dithering method specified by dmode.


+++ chroma +++

True to process both Luma and Chroma planes, False to process only Luma. Default: True


+++ Advanced Settings +++

gpuid = 0, the GPU id to use for KNLMeans and BM3D  
d = 2, KNLMeans 'd' parameter, can be set to 3 for small quality improvement

Normally you shouldn't have to touch these  
rgmode = 18, RemoveGrain mode used during post-processing. Setting this to 0 disables post-processing, useful to compare raw denoising.  
thsad = 400, threshold used for MVTools analysis  
a = 2, KNLMeans 'a' parameter  
h = 1.4, KNLMeans 'h' parameter  
sigma = 9, BM3D 'sigma' parameter


Deband and dithering should always be applied at the very end of your script. If you plan to do further processing, disable deband and using ordered dithering method.

TODO  
- YUV420 or YUV444 output?  
- veed, autolevels... include or not?  
- Allow disabling GPU acceleration for both KNLMeans and BM3D, separately  
- Support GRAY formats and test chroma=False
