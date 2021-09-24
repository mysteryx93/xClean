# xClean
xClean spatio/temporal denoiser for VapourSynth/Avisynth

Based on [mClean by burfadel](https://forum.doom9.org/showthread.php?t=174804)

Mod by Etienne Charland:
- Added dynamic noise reduction strength based on Luma where dark areas get full reduction and 
white areas preserve more of the source. Set Strength between 0 and -200, recommended -50. A value of -50 means that out of 255 values (or 219 tv range), 
the 50 blackest values have full-reduction and the 50 whitest values are merged at a minimal strength of 50/255.
- Strength no longer apply to deband and sharpen, only to noise reduction.
- Deband was denoised and then sharpened. It has been moved to the end after sharpening.
- Veed is run between noise reduction and sharpening and is not affected by strength.
- Added boost denoising to denoise dark scenes using a different method (KNLMeansCL).
- Now support various denoising methods: MvTools2 (default) and KNLMeansCL (default for boost)
- thSAD parameter has been replaced by p1, which sets thSAD for MvTools2 and h for KNLMeansCL
- Can now run KNLMeansCL with renoise and sharpen using method=1

Requires: rgsv, rgvs, fmtc, vcm, mv, mvsf, tmedian

## Description  
Typical spatial filters work by removing large variations in the image on a small scale, reducing noise but also making the image less
sharp or temporally stable. xClean removes noise whilst retaining as much detail as possible, as well as provide optional image enhancement.

xClean works primarily in the temporal domain, although there is some spatial limiting.
Chroma is processed a little differently to luma for optimal results.
Chroma processing can be disabled with chroma = False.

+++ Artifacts +++  
Spatial picture artifacts may remain as removing them is a fine balance between removing the unwanted artifact whilst not removing detail.
Additional dering/dehalo/deblock filters may be required, but should ONLY be uses if required due the detail loss/artifact removal balance.

+++ Sharpening +++  
Applies a modified unsharp mask to edges and major detected detail. Range of normal sharpening is 0-20. There are 4 additional settings,
21-24 that provide 'overboost' sharpening. Overboost sharpening is only suitable typically for high definition, high quality sources.
Actual sharpening calculation is scaled based on resolution.

+++ ReNoise +++  
ReNoise adds back some of the removed luma noise. Re-adding original noise would be counterproductive, therefore ReNoise modifies this noise
both spatially and temporally. The result of this modification is the noise becomes much nicer and it's impact on compressibility is greatly
reduced. It is not applied on areas where the sharpening occurs as that would be counterproductive. Settings range from 0 to 20.
The strength of renoise is affected by the the amount of original noise removed and how this noise varies between frames.
It's main purpose is to reduce the 'flatness' that occurs with any form of effective denoising.

+++ Deband +++  
This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. The settings are not adjustable as the default settings are suitable for most cases without having a large effect
on compressibility. 0 = disabled, 1 = deband only, 2 = deband and veed

+++ Depth +++  
This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

+++ Strength +++  
The strength of the denoising effect can be adjusted using this parameter. It ranges from 20 percent denoising effect with strength 0, up to the
100 percent of the denoising with strength 20. This function works by blending a scaled percentage of the original image with the processed image.
A value between 0 and -200 will apply dynamic noise reduction strength based on Luma, where black zones get full denoising and white areas
preserve the source. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction and the 50 whitest values 
are merged at a minimal strength of 50/255.

+++ Boost +++  
Boost denoising using a secondary denoiser for dark scenes, between 0 and 100. 0 to disable.
Setting a value of 5 means that frames with average luma below 5% will be merged between method and methodboost.
It will merge at a ratio of .8 - luma / boost * .8

+++ Method / MethodBoost +++  
0 for MvTools (default), 1 for KNLMeansCL (default for dark scene boost)

+++ Outbits +++  
Specifies the bits per component (bpc) for the output for processing by additional filters. It will also be the bpc that xClean will process.
If you output at a higher bpc keep in mind that there may be limitations to what subsequent filters and the encoder may support.

+++ p1 +++  
Parameter to configure denoising method.  
When method=0, sets thSAD for MvTools2 analysis. Default=400  
When method=1, sets the h (strength) parameter of KNLMeansCL. Default=4.0

+++ b1 +++  
Parameter to configure boost denoising method.  
When method=0, sets thSAD for MvTools2 analysis. Default=400  
When method=1, sets the h (strength) parameter of KNLMeansCL. Default=8.0
