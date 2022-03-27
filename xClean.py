from vapoursynth import core
import vapoursynth as vs
import math
from typing import Optional
import nnedi3_resample as nnedi3

"""
xClean 3-pass denoiser
beta 8 (2021-11-10) by Etienne Charland
Supported formats: YUV, RGB, GRAY
Requires: rgsf, rgvs, fmtc, mv, mvsf, tmedian, knlm, bm3d, bm3dcuda_rtc, bm3dcpu, neo_f3kdb, akarin, nnedi3_resample, nnedi3cl

xClean runs MVTools -> BM3D -> KNLMeans in that order, passing the output of each pass as the ref of the next denoiser.

The objective is to remove noise while preserving as much details as possible. Removing noise is easy -- just blur out everything.
The hard work is in preserving the details in a way that feels natural.

Designed for raw camera footage to remove noise in dark areas while preserving the fine details. It works for most types of content.

Performance-wise, BM3D pass is the heaviest and helps recover fine details, but this script runs 1 pass of BM3D whereas stand-alone BM3D runs twice.


+++ Short Doc (TL;DR) +++
Default settings provide the best quality in most cases. Simply use
xClean(sharp=..., outbits=...)

If only darker areas contain noise, set strength=-50
For better performance, set m1=0 or m2=0, or set m1=.5 and m2=3.6 (downscale)
BM3D performance can be greatly improved by setting radius=0, block_step=7, bm_range=7, ps_range=5

For 720p WebCam, optimal settings are: sharp=9.5, m1=.65, h=2.8
For 288p anime, optimal settings are: sharp=9.5, m1=.7, rn=0, optional depth=1
For 4-5K GoPro (with in-camera sharpening at Low), optimal settings are: sharp=7.7, m1=.5, m2=3.7, optional strength=-50 (or m1=.6, m2=3.8 if your computer can handle it)


+++ Description +++
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


+++ Denoising Methods Overview +++
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


+++ Denoising Pass Configuration  (m1=.6, m2=2, m3=2) +++
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


+++ Renoise and Sharpen  (rn=14, sharp=9.5) +++
The idea comes from mClean by Burfadel (https://forum.doom9.org/showthread.php?t=174804) and the algorithm was changed by someone else while porting 
to VapourSynth, producing completely different results -- original Avisynth version blurs a lot more, VapourSynth version keeps a lot more details.

It may sound counter-productive at first, but the idea is to combat the flat or plastic effect of denoising by re-introducing part of the removed noise.
The noise is processed and stabilized before re-inserting so that it's less distracting.
Renoise also helps reduce large-radius grain; but should be disabled for anime (rn=0).

Using the same analysis data, it's also sharpening to compensate for denoising blur.
Sharpening must be between 0 and 20. Actual sharpening calculation is scaled based on resolution.


+++ Strength / Dynamic Denoiser Strength  (strength=20) +++
A value of 20 will denoise normally.
A value between 1 and 19 will reduce the denoising effect by partially merging back with the original clip.
A value between 0 and -200 will activate Dynamic Denoiser Strength, useful when bright colors require little or no denoising and dark colors contain more noise.
It applies a gradual mask based on luma. Specifying a value of -50 means that out of 255 (or 219 tv range), the 50 blackest values have full-reduction 
and the 50 whitest values are merged at a minimal strength of 50/255 = 20%.

+++ Radius  (radius=0) +++
BM3D radius. Low impact on individual frames.
Pros: Helps stabilize temporal grain. Can significantly improve video compressability.
Cons: High impact on performance and memory usage! May require downscaling BM3D for HD content with m2 between 3.6 and 3.8
For moving water, the temporal stabilization may be undesirable.

+++ Depth  (depth=0) +++
This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

+++ Deband  (deband=False) +++
This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
to both luma and chroma. Default settings are suitable for most cases without having a large effect on compressibility.

+++ Output  (outbits, dmode=0) +++
Specifies the output bitdepth. If not specified it will be converted back to the bitdepth of the source clip using dithering method specified by dmode.
You can set dmode=3 if you won't be doing any further processing for high-quality ditherig.

+++ Chroma upsampling/downsamping  (chroma=nnedi3, downchroma=True) +++
Chroma upsampling options:
none = don't touch chroma
bicubic = bicubic(0, .5) upsampling
nnedi3 = NNEDI3 upsampling
reconstructor = feisty2's ChromaReconstructor_faster v3.0 HBD mod

downchroma: whether to downscale back to match source clip. Default is False for reconstructor and True for other methods.

+++ Anime +++
For anime, set rn=0. Optionally, you can set depth to 1 or 2 to thicken the lines.

+++ Advanced Settings +++
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
"""

def xClean(clip: vs.VideoNode, chroma: str = "nnedi3", sharp: float = 9.5, rn: float = 14, deband: bool = False, depth: int = 0, strength: int = 20, m1: float = .6, m2: int = 2, m3: int = 2, outbits: Optional[int] = None,
        dmode: int = 0, rgmode: int = 18, thsad: int = 400, d: int = 2, a: int = 2, h: float = 1.4, gpuid: int = 0, gpucuda: Optional[int] = None, sigma: float = 9, 
        block_step: int = 4, bm_range: int = 16, ps_range: int = 8, radius: int = 0, bm3d_fast: bool = False, conv: bool = True, downchroma: bool = None) -> vs.VideoNode:

    width = clip.width
    height = clip.height
    defH = max(height, width // 4 * 3) # Resolution calculation for auto blksize settings
    if sharp < 0 or sharp > 20:
        raise ValueError("xClean: sharp must be between 0 and 20")
    if rn < 0 or rn > 20:
        raise ValueError("xClean: rn (renoise strength) must be between 0 and 20")
    if depth < 0 or depth > 5:
        raise ValueError("xClean: depth must be between 0 and 5")
    if strength < -200 or strength > 20:
        raise ValueError("xClean: strength must be between -200 and 20")
    if m1 < 0 or m1 >= 4:
        raise ValueError(r"xClean: m1 (MVTools pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    if m2 < 0 or m2 >= 4:
        raise ValueError("xClean: m2 (BM3D pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    if m3 < 0 or m3 > 3:
        raise ValueError("xClean: m3 (KNLMeansCL pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit), 3 (32-bit)")
    if m1 == 0 and m2 == 0 and m3 == 0:
        raise ValueError("xClean: At least one pass must be enabled")
    if not chroma in ["none", "bicubic", "nnedi3", "reconstructor"]:
        raise ValueError("xClean: chroma must be none, bicubic, nnedi3 or reconstructor")

    uv = clip
    if chroma == "none":
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    
    samp = ClipSampling(clip)
    isGray = samp == "GRAY"
    if isGray:
        chroma = "none"
        conv = False
    dochroma = chroma != "none" or samp == "RGB"
    downchroma = downchroma or False if chroma == "reconstructor" else True

    gpucuda = gpucuda if gpucuda != None else gpuid
    bd = clip.format.bits_per_sample
    fulls = GetColorRange(clip) == 0
    matrix = GetMatrix(clip)
    outbits = outbits or bd
    if not outbits in [8, 9, 10, 12, 14, 16, 32]:
        raise ValueError("xClean: outbits must be 8, 9, 10, 12, 14, 16 or 32")
    cplace = ["left", "center", "top_left", "left", "left", "left"] [GetChromaLoc(clip)]

    # Reference clips are in RGB or GRAY format, to allow converting to desired formats
    cconv = ConvertBits(clip, 16, fulls, True) if bd < 16 else clip
    cconv = cconv if samp in ["444", "RGB", "GRAY"] else \
        ChromaReconstructor(cconv, gpuid) if chroma == "reconstructor" else \
        nnedi3.nnedi3_resample(cconv, csp=vs.YUV444P16 if bd < 32 else vs.YUV444PS, mode="nnedi3cl" if gpuid >= 0 else "znedi3", device=max(0, gpuid), fulls=fulls, fulld=fulls) if chroma == "nnedi3" else \
        core.fmtc.resample(cconv, csp=vs.YUV444P16 if bd < 32 else vs.YUV444PS, kernel="bicubic", a1=0, a2=.5, fulls=fulls, fulld=fulls, cplace=cplace)
    cconv = ConvertMatrix(cconv, vs.RGB, fulls) if conv and clip.format.color_family == vs.YUV else cconv
    c32 = ConvertBits(cconv, 32, fulls, False)
    c16 = ConvertBits(cconv, 16, fulls, True)
    c8 = ConvertBits(cconv, 8, fulls, True)
    output = None

    # Apply MVTools
    if m1 > 0:
        m1r = 1 if m1 == int(m1) else m1 % 1 # Decimal point is resize factor
        m1 = int(m1)
        c1 = c32 if m1 == 3 else c16 if m1 == 2 else c8
        c1 = c1.fmtc.resample((width * m1r)//4*4, (height * m1r)//4*4, kernel="bicubic", a1=0, a2=.75) if m1r < 1 else c1
        c1 = RGB_to_YCgCoR(c1, fulls) if conv else c1
        output = MvTools(c1, defH, thsad)
        sharp1 = max(0, min(20, sharp + (1 - m1r) * .35))
        output = PostProcessing(output, c1, defH, strength, sharp1, rn, rgmode, 0)
        # output in YCgCoR format

    # Apply BM3D
    if m2 > 0:
        m2r = 1 if m2 == int(m2) else m2 % 1 # Decimal point is resize factor
        m2 = int(m2)
        m2o = max(2, max(m2, m3))
        c2 = c32 if m2o==3 else c16
        ref = RGB_to_OPP(YCgCoR_to_RGB(output, fulls), fulls) if output and conv else output if output else None
        ref = ref.fmtc.resample((width * m2r)//4*4, (height * m2r)//4*4, csp = vs.GRAYS if isGray else vs.YUV444PS, kernel = "spline36") if ref else None
        c2r = c2.fmtc.resample((width * m2r)//4*4, (height * m2r)//4*4, kernel = "bicubic", a1=0, a2=0.5) if m2r < 1 else c2
        c2r = ConvertBits(RGB_to_OPP(c2r, fulls) if conv else c2r, 32, fulls, False)

        output = BM3D(c2r, ref, sigma, gpucuda, block_step, bm_range, ps_range, radius, bm3d_fast)
        
        output = ConvertBits(output, c2.format.bits_per_sample, fulls, False)
        output = RGB_to_YCgCoR(OPP_to_RGB(output, fulls), fulls) if conv else output
        c2 = RGB_to_YCgCoR(c2, fulls) if conv else c2
        output = output.fmtc.resample(width, height, kernel = "spline36") if m2r < 1 else output
        sharp2 = max(0, min(20, sharp + (1 - m2r) * .95))
        output = PostProcessing(output, c2, defH, strength, sharp2, rn, rgmode, 1)
        # output in YCgCoR format

    if output and output.height < height:
        output = output.fmtc.resample(width, height, kernel = "spline36")

    # Apply KNLMeans
    if m3 > 0:
        m3 = min(2, m3) # KNL internally computes in 16-bit
        c3 = c32 if m3==3 else c16
        c3 = RGB_to_YCgCoR(c3, fulls) if conv else c3
        ref = ConvertBits(output, c3.format.bits_per_sample, fulls, False) if output else None
        output = KnlMeans(c3, ref, d, a, h, gpuid)
        # Adjust sharp based on h parameter.
        sharp3 = max(0, min(20, sharp - .5 + (h/2.8)))
        output = PostProcessing(output, c3, defH, strength, sharp3, rn, rgmode, 2)
        # output in YCgCoR format

    # Add Depth (thicken lines for anime)
    if depth:
        depth2 = -depth*3
        depth = depth*2
        output = core.std.MergeDiff(output, core.std.MakeDiff(output.warp.AWarpSharp2(128, 3, 1, depth2, 1), output.warp.AWarpSharp2(128, 2, 1, depth, 1)))
    
    # Apply deband
    if deband:
        if output.format.bits_per_sample > 16:
            output = ConvertBits(output, 16, fulls, False)
        output = output.neo_f3kdb.Deband(range=16, preset="high" if dochroma else "luma", grainy=defH/15, grainc=defH/16 if dochroma else 0)

    # Convert to desired output format and bitrate
    output = YCgCoR_to_RGB(output, fulls) if conv else output
    if clip.format.color_family == vs.YUV:
        output = ConvertMatrix(output, vs.YUV, fulls, matrix)
        if downchroma and samp != "444":
            output = output.fmtc.resample(css=samp, cplace=cplace, fulls=fulls, fulld=fulls, kernel="bicubic", a1=0, a2=0.5)
    if output.format.bits_per_sample != outbits:
        output = output.fmtc.bitdepth(bits=outbits, fulls=fulls, fulld=fulls, dmode=dmode)
     
    # Merge source chroma planes if not processing chroma.
    if not dochroma and uv.format.color_family == vs.YUV:
        if uv.format.bits_per_sample != outbits:
            uv = ConvertBits(uv, outbits, fulls, True)
        output = core.std.ShufflePlanes([output, uv], [0, 1, 2], vs.YUV)
    
    return output


def PostProcessing(clean: vs.VideoNode, c: vs.VideoNode, defH: int, strength: int, sharp: float, rn: float, rgmode: int, method: int) -> vs.VideoNode:
    fulls = GetColorRange(c) == 0
    if rgmode == 0:
        sharp = rn = 0

    # Run at least in 16-bit
    if clean.format.bits_per_sample < 16:
        clean = ConvertBits(clean, 16, fulls, False)
    if c.format.bits_per_sample < 16:
        c = ConvertBits(c, 16, fulls, False)
    bd = clean.format.bits_per_sample
    
    # Separate luma and chroma
    filt = clean
    clean = core.std.ShufflePlanes(clean, [0], vs.GRAY)
    cy = core.std.ShufflePlanes(c, [0], vs.GRAY)

    # Spatial luma denoising
    RG = core.rgsf.RemoveGrain if bd == 32 else core.rgvs.RemoveGrain
    clean2 = RG(clean, rgmode) if rgmode > 0 else clean

    # Apply dynamic noise reduction strength based on Luma
    if strength <= 0:
        # Slightly widen the exclusion mask to preserve details and edges
        cleanm = cy.std.Maximum()
        if defH > 500:
            cleanm = cleanm.std.Maximum()
        if defH > 1200:
            cleanm = cleanm.std.Maximum()

        # Adjust mask levels
        cleanm = cleanm.std.Levels((0 if fulls else 16) - strength, 255 if fulls else 235, 0.85, 0, 255+strength)

        # Merge based on luma mask
        clean = core.std.MaskedMerge(clean, cy, cleanm)
        clean2 = core.std.MaskedMerge(clean2, cy, cleanm)
        filt = core.std.MaskedMerge(filt, c, cleanm)
    elif strength < 20:
        # Reduce strength by partially merging back with original
        clean = core.std.Merge(cy, clean, 0.2+0.04*strength)
        clean2 = core.std.Merge(cy, clean2, 0.2+0.04*strength)
        filt = core.std.Merge(c, filt, 0.2+0.04*strength)

    # Unsharp filter for spatial detail enhancement
    if sharp:
        RE = core.rgsf.Repair if bd == 32 else core.rgvs.Repair
        mult = .69 if method == 2 else .14 if method == 1 else 1
        sharp = min(50, (15 + defH * sharp * 0.0007) * mult)
        clsharp = core.std.MakeDiff(clean, Sharpen(clean2, amountH=-0.08-0.03*sharp))
        clsharp = core.std.MergeDiff(clean2, RE(clsharp.tmedian.TemporalMedian(), clsharp, 12))
    
    # If selected, combining ReNoise
    noise_diff = core.std.MakeDiff(clean2, cy)

    if rn:
        i = 0.00392 if bd == 32 else 1 << (bd - 8)
        peak = 1.0 if bd == 32 else (1 << bd) - 1
        expr = "x {a} < 0 x {b} > {p} 0 x {c} - {p} {a} {d} - / * - ? ?".format(a=32*i, b=45*i, c=35*i, d=65*i, p=peak)
        clean1 = core.std.Merge(clean2, core.std.MergeDiff(clean2, Tweak(noise_diff.tmedian.TemporalMedian(), cont=1.008+0.00016*rn)), 0.3+rn*0.035)
        clean2 = core.std.MaskedMerge(clean2, clean1, core.std.Expr([core.std.Expr([clean, clean.std.Invert()], 'x y min')], [expr]))

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.std.Binarize().std.Invert()
    if rgmode > 0:
        clean2 = core.std.MaskedMerge(clean2, clsharp if sharp else clean, core.std.Expr([noise_diff, clean.std.Sobel()], 'x y max'))

    # Combining result of luma and chroma cleaning
    return core.std.ShufflePlanes([clean2, filt], [0, 1, 2], vs.YUV) if c.format.color_family == vs.YUV else clean2


# mClean denoising method
def MvTools(c: vs.VideoNode, defH: int, thSAD: int) -> vs.VideoNode:
    bd = c.format.bits_per_sample
    fulls = GetColorRange(c) == 0
    icalc = bd < 32
    S = core.mv.Super if icalc else core.mvsf.Super
    A = core.mv.Analyse if icalc else core.mvsf.Analyse
    R = core.mv.Recalculate if icalc else core.mvsf.Recalculate

    sc = 8 if defH > 2880 else 4 if defH > 1440 else 2 if defH > 720 else 1
    bs = 16 if defH / sc > 360 else 8
    ov = 6 if bs > 12 else 2
    pel = 1 if defH > 720 else 2
    lampa = 777 * (bs ** 2) // 64
    truemotion = False if defH > 720 else True

    ref = c.std.Convolution(matrix=[2, 3, 2, 3, 6, 3, 2, 3, 2])
    super1 = S(ref, hpad=bs, vpad=bs, pel=pel, rfilter=4, sharp=1)
    super2 = S(c, hpad=bs, vpad=bs, pel=pel, rfilter=1, levels=1)
    analyse_args = { 'blksize': bs, 'overlap': ov, 'search': 5, 'truemotion': truemotion }
    recalculate_args = { 'blksize': bs, 'overlap': ov, 'search': 5, 'truemotion': truemotion, 'thsad': 180, 'lambda': lampa }

    # Analysis
    bvec4 = R(super1, A(super1, isb=True,  delta=4, **analyse_args), **recalculate_args) if not icalc else None
    bvec3 = R(super1, A(super1, isb=True,  delta=3, **analyse_args), **recalculate_args)
    bvec2 = R(super1, A(super1, isb=True,  delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    bvec1 = R(super1, A(super1, isb=True,  delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec1 = R(super1, A(super1, isb=False, delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec2 = R(super1, A(super1, isb=False, delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    fvec3 = R(super1, A(super1, isb=False, delta=3, **analyse_args), **recalculate_args)
    fvec4 = R(super1, A(super1, isb=False, delta=4, **analyse_args), **recalculate_args) if not icalc else None

    # Applying cleaning
    if icalc:
        clean = core.mv.Degrain3(c, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=thSAD)
    else:
        clean = core.mvsf.Degrain4(c, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)

    if bd < 16:
        clean = ConvertBits(clean, 16, fulls, False)
        c = ConvertBits(c, 16, fulls, False)

    if c.format.color_family == vs.YUV:
        uv = core.std.MergeDiff(clean, core.tmedian.TemporalMedian(core.std.MakeDiff(c, clean, [1, 2]), 1, [1, 2]), [1, 2])
        clean = core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)
    return clean


# BM3D denoising method
def BM3D(clip: vs.VideoNode, ref: Optional[vs.VideoNode], sigma: float, gpuid: int, block_step: int, bm_range: int, ps_range: int, radius: int, bm3d_fast: bool) -> vs.VideoNode:
    matrix = GetMatrix(clip)
    fulls = GetColorRange(clip)
    chroma = clip.format.color_family==vs.YUV
    icalc = clip.format.bits_per_sample < 32
    if gpuid >= 0:
        clean = core.bm3dcuda_rtc.BM3D(clip, ref, chroma=chroma, sigma=sigma, device_id=gpuid, fast=bm3d_fast, radius=radius, block_step=block_step, bm_range=bm_range, ps_range=ps_range)
    else:
        clean = core.bm3dcpu.BM3D(clip, ref, chroma=chroma, sigma=sigma, block_step=block_step, bm_range=bm_range, ps_range=ps_range, radius=radius)
    clean = clean.bm3d.VAggregate(sample=0 if icalc else 1, radius=radius) if radius > 0 else clean
    return clean


# KnlMeansCL denoising method, useful for dark noisy scenes
def KnlMeans(clip: vs.VideoNode, ref: Optional[vs.VideoNode], d: int, a: int, h: float, gpuid: int) -> vs.VideoNode:
    #if ref and ref.format != clip.format:
    #    ref = ref.resize.Bicubic(format=clip.format)
    bd = clip.format.bits_per_sample
    fulls = GetColorRange(clip) == 0
    if ref and ref.format.bits_per_sample != bd:
        ref = ConvertBits(ref, bd, fulls, True)
    src = clip
    sample = 1 if bd == 32 else 0

    device = dict(device_type="auto" if gpuid >= 0 else "cpu", device_id=max(0, gpuid))
    if clip.format.color_family == vs.GRAY:
        output = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, **device)
    elif ClipSampling(clip) == "444":
        output = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="YUV", rclip=ref, **device)
    else:
        clean = clip.knlm.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, **device)
        uv = clip.knlm.KNLMeansCL(d=d, a=a, h=h/2, channels="UV", rclip=ref, **device)
        output = core.std.ShufflePlanes(clips=[clean, uv], planes=[0, 1, 2], colorfamily=vs.YUV)
    return output


# Adjusts brightness and contrast
def Tweak(clip: vs.VideoNode, bright: float = None, cont: float = None) -> vs.VideoNode:
    fulls = GetColorRange(clip) == 0
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if clip.format.color_family == vs.RGB:
        raise TypeError("Tweak: RGB color family is not supported!")

    if not (bright is None and cont is None):
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if isFLOAT:
            expr = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)
            clip =  core.std.Expr([clip], [expr] if isGRAY else [expr, ''])
        else:
            luma_lut = []
            luma_min = 16  << (bd - 8) if not fulls else 0
            luma_max = 235 << (bd - 8) if not fulls else (1 << bd) - 1

            for i in range(1 << bd):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = core.std.Lut(clip, [0], luma_lut)
    return clip


# from muvsfunc
def Sharpen(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None, planes = None) -> vs.VideoNode:
    # Avisynth's internel filter Sharpen()
    funcName = 'Sharpen'
    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    planes = list(range(clip.format.num_planes)) if planes is None else planes

    center_weight_v = math.floor(2 ** (amountV - 1) * 1023 + 0.5)
    outer_weight_v = math.floor((0.25 - 2 ** (amountV - 2)) * 1023 + 0.5)
    center_weight_h = math.floor(2 ** (amountH - 1) * 1023 + 0.5)
    outer_weight_h = math.floor((0.25 - 2 ** (amountH - 2)) * 1023 + 0.5)

    conv_mat_v = [outer_weight_v, center_weight_v, outer_weight_v]
    conv_mat_h = [outer_weight_h, center_weight_h, outer_weight_h]

    if math.fabs(amountH) >= 0.00002201361136: # log2(1+1/65536)
        clip = core.std.Convolution(clip, conv_mat_v, planes=planes, mode='v')

    if math.fabs(amountV) >= 0.00002201361136:
        clip = core.std.Convolution(clip, conv_mat_h, planes=planes, mode='h')
    return clip


# Point resize is 1.5x faster than fmtc
def ConvertBits(c: vs.VideoNode, bits: int = 8, fulls: bool = False, dither: bool = False):
    if c.format.bits_per_sample == bits:
        return c
    return c.fmtc.bitdepth(bits=bits, fulls=fulls, fulld=fulls, dmode=0 if dither else 1)


def GetFormat(color_family: int, bits: int, sampw: int = 0, samph: int = 0):
    return core.query_video_format(
                            color_family    = color_family,
                            sample_type     = vs.FLOAT if bits==32 else vs.INTEGER,
                            bits_per_sample = bits,
                            subsampling_w   = sampw,
                            subsampling_h   = samph)


def ClipSampling(clip: vs.VideoNode) -> str:
    return "GRAY" if clip.format.color_family == vs.GRAY else \
            "RGB" if clip.format.color_family == vs.RGB else \
            ("444" if clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0 else \
            "422" if clip.format.subsampling_w == 1 and clip.format.subsampling_h == 0 else "420") \
            if clip.format.color_family == vs.YUV else "UNKNOWN"


# Get frame properties
def GetFrameProp(c: vs.VideoNode, name: str, default):
    props = c.get_frame(0).props
    return props[name] if name in props else default

def GetColorRange(c: vs.VideoNode) -> int:
    return 0 if c.format.color_family == vs.RGB else GetFrameProp(c, "_ColorRange", 1)

def GetMatrix(c: vs.VideoNode) -> int:
    if c.format.color_family == vs.RGB:
        return 0
    matrix = GetFrameProp(c, "_Matrix", 1)
    return 6 if matrix in [0, 2] else matrix

def GetTransfer(c: vs.VideoNode) -> int:
    transfer = GetFrameProp(c, "_Transfer", 0)
    return GetMatrix(c) if transfer in [0, 2] else transfer

def GetPrimaries(c: vs.VideoNode) -> int:
    primaries = GetFrameProp(c, "_Primaries", 0)
    return GetMatrix(c) if primaries in [0, 2] else primaries

def GetChromaLoc(c: vs.VideoNode) -> int:
    return GetFrameProp(c, "_ChromaLocation", 0)


# Converts matrix into desired format. If matrix is not specified, it will read matrix from source frame property.
def ConvertMatrix(c: vs.VideoNode, col_fam: int, fulls: bool, matrix: Optional[int] = None):
    matrix = matrix if matrix != None else GetMatrix(c)
    csp = GetFormat(col_fam, c.format.bits_per_sample)
    if matrix == 10:
        return c.fmtc.matrix2020cl(csp=csp, full=fulls)
    else:
        mat = ["RGB", "709", "601", None, "FCC", "601", "601", "240", "YCgCo", "2020", None, None, None, None] [matrix]
        if mat == None:
            raise ValueError(f"ConvertMatrix: matrix {matrix} is not supported.")
        return c.fmtc.matrix(csp=csp, mat=mat, fulls=fulls, fulld=fulls)


# RGB to YCgCo RCT function
def RGB_to_YCgCoR (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.RGB:
        raise TypeError("RGB_to_YCgCoR: Clip is not in RGB format!")

    bd = c.format.bits_per_sample
    R = core.std.ShufflePlanes(c, [0], vs.GRAY)
    G = core.std.ShufflePlanes(c, [1], vs.GRAY)
    B = core.std.ShufflePlanes(c, [2], vs.GRAY)

    Co = core.std.Expr([R,      B], ex_dlut("x 0.5  * y 0.5  * - range_half +",                bd, fulls))
    Cg = core.std.Expr([Co, G,  B], ex_dlut("y z x range_half - 0.5 * + - 0.5 * range_half +", bd, fulls))
    Y  = core.std.Expr([Co, Cg, B], ex_dlut("z x range_half - 0.5 * + y range_half - +",       bd, fulls))

    output = core.std.ShufflePlanes([Y, Cg, Co], [0, 0, 0], vs.YUV)
    return output.std.SetFrameProp(prop='_Matrix', intval=2)


#  YCgCo RCT to RGB function
def YCgCoR_to_RGB (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.YUV:
        raise TypeError("YCgCoR_to_RGB: Clip is not in YUV format!")

    bd = c.format.bits_per_sample
    Y = core.std.ShufflePlanes(c, [0], vs.GRAY)
    Cg = core.std.ShufflePlanes(c, [1], vs.GRAY)
    Co = core.std.ShufflePlanes(c, [2], vs.GRAY)

    G = core.akarin.Expr([Y, Cg    ], ex_dlut("y range_half - dup yvar! 2 * x yvar@ - +",             bd, fulls))
    B = core.std.Expr([Y, Cg, Co], ex_dlut("x y range_half - - z range_half - 0.5 * -", bd, fulls))
    R = core.std.Expr([Co, B    ], ex_dlut("y x range_half - 2 * +",                    bd, fulls))

    output = core.std.ShufflePlanes([R, G, B], [0, 0, 0], vs.RGB)
    return output.std.SetFrameProp(prop='_Matrix', intval=0)


def RGB_to_OPP (c: vs.VideoNode, fulls: bool = False) -> vs.VideoNode:
    if c.format.color_family != vs.RGB:
        raise TypeError("RGB_to_YCgCoR: Clip is not in RGB format!")

    bd = c.format.bits_per_sample
    R = core.std.ShufflePlanes(c, [0], vs.GRAY)
    G = core.std.ShufflePlanes(c, [1], vs.GRAY)
    B = core.std.ShufflePlanes(c, [2], vs.GRAY)

    b32 = "" if bd == 32 else "range_half +"

    O  = core.std.Expr([R, G, B], ex_dlut("x y z + + 0.333333333 *",     bd, fulls))
    P1 = core.std.Expr([R,    B], ex_dlut("x y - 0.5 * "+b32,            bd, fulls))
    P2 = core.std.Expr([R, G, B], ex_dlut("x z + 0.25 * y 0.5 * - "+b32, bd, fulls))

    output = core.std.ShufflePlanes([O, P1, P2], [0, 0, 0], vs.YUV)
    return output.std.SetFrameProp(prop='_Matrix', intval=2)


def OPP_to_RGB (c: vs.VideoNode, fulls: bool = False):
    if c.format.color_family != vs.YUV:
        raise TypeError("YCgCoR_to_RGB: Clip is not in YUV format!")

    bd = c.format.bits_per_sample
    O = core.std.ShufflePlanes(c, [0], vs.GRAY)
    P1 = core.std.ShufflePlanes(c, [1], vs.GRAY)
    P2 = core.std.ShufflePlanes(c, [2], vs.GRAY)

    b32 = "" if bd == 32 else "range_half -"

    R = core.std.Expr([O, P1, P2], ex_dlut("x y "+b32+" + z "+b32+" 0.666666666 * +", bd, fulls))
    G = core.std.Expr([O,     P2], ex_dlut("x y "+b32+" 1.333333333 * -",             bd, fulls))
    B = core.std.Expr([O, P1, P2], ex_dlut("x z "+b32+" 0.666666666 * + y "+b32+" -", bd, fulls))

    output = core.std.ShufflePlanes([R, G, B], [0, 0, 0], vs.RGB)
    return output.std.SetFrameProp(prop='_Matrix', intval=0)


# HBD constants 3D look up table
#
# * YUV and RGB mid-grey is 127.5 (rounded to 128) for PC range levels,
#   this translates to a value of 125.5 in TV range levels. Chroma is always centered, so 128 regardless.
def ex_dlut(expr: str = "", bits: int = 8, fulls: bool = False) -> str:
    bitd = \
        0 if bits == 8 else \
        1 if bits == 10 else \
        2 if bits == 12 else \
        3 if bits == 14 else \
        4 if bits == 16 else \
        5 if bits == 24 else \
        6 if bits == 32 else -1
    if bitd < 0:
        raise ValueError(f"ex_dlut: Unsupported bit depth ({bits})")
    
    #                 8-bit UINT      10-bit UINT          12-bit UINT          14-bit UINT            16-bit UINT         24-bit UINT               32-bit Ufloat
    range_min   = [  (  0.,  0.),    (   0.,   0.   ),    (   0.,   0.   ),    (    0.,    0.   ),    (    0.,    0.),    (       0.,       0.),    (       0.,       0.)   ]   [bitd]
    ymin        = [  ( 16., 16.),    (  64.,  64.   ),    ( 256., 257.   ),    ( 1024., 1028.   ),    ( 4096., 4112.),    ( 1048576., 1052672.),    (  16/255.,  16/255.)   ]   [bitd]
    cmin        = [  ( 16., 16.),    (  64.,  64.   ),    ( 256., 257.   ),    ( 1024., 1028.   ),    ( 4096., 4112.),    ( 1048576., 1052672.),    (  16/255.,  16/255.)   ]   [bitd]
    ygrey       = [  (126.,126.),    ( 502., 504.   ),    (2008.,2016.   ),    ( 8032., 8063.   ),    (32128.,32254.),    ( 8224768., 8256896.),    ( 125.5/255.,125.5/255.)]   [bitd]
    range_half  = [  (128.,128.),    ( 512., 514.   ),    (2048.,2056.   ),    ( 8192., 8224.   ),    (32768.,32896.),    ( 8388608., 8421376.),    ( 128/255., 128/255.)   ]   [bitd]
    yrange      = [  (219.,219.),    ( 876., 879.   ),    (3504.,3517.688),    (14016.,14070.750),    (56064.,56283.),    (14352384.,14408448.),    ( 219/255., 219/255.)   ]   [bitd]
    crange      = [  (224.,224.),    ( 896., 899.500),    (3584.,3598.   ),    (14336.,14392.   ),    (57344.,57568.),    (14680064.,14737408.),    ( 224/255., 224/255.)   ]   [bitd]
    ymax        = [  (235.,235.),    ( 940., 943.672),    (3760.,3774.688),    (15040.,15098.750),    (60160.,60395.),    (15400960.,15461120.),    ( 235/255., 235/255.)   ]   [bitd]
    cmax        = [  (240.,240.),    ( 960., 963.750),    (3840.,3855.   ),    (15360.,15420.   ),    (61440.,61680.),    (15728640.,15790080.),    ( 240/255., 240/255.)   ]   [bitd]
    range_max   = [  (255.,255.),    (1020.,1023.984),    (4080.,4095.938),    (16320.,16383.750),    (65280.,65535.),    (16711680.,16776960.),    (       1.,       1.)   ]   [bitd]
    range_size  = [  (256.,256.),    (1024.,1024.   ),    (4096.,4096.   ),    (16384.,16384.   ),    (65536.,65536.),    (16777216.,16777216.),    (       1.,       1.)   ]   [bitd]

    fs  = 1 if fulls else 0
    expr = expr.replace("ymax ymin - range_max /", str(yrange[fs]/range_max[fs]))
    expr = expr.replace("cmax cmin - range_max /", str(crange[fs]/range_max[fs]))
    expr = expr.replace("cmax ymin - range_max /", str(crange[fs]/range_max[fs]))
    expr = expr.replace("range_max ymax ymin - /", str(range_max[fs]/yrange[fs]))
    expr = expr.replace("range_max cmax cmin - /", str(range_max[fs]/crange[fs]))
    expr = expr.replace("range_max cmax ymin - /", str(range_max[fs]/crange[fs]))
    expr = expr.replace("ymax ymin -",             str(yrange[fs]))
    expr = expr.replace("cmax ymin -",             str(crange[fs]))
    expr = expr.replace("cmax cmin -",             str(crange[fs]))

    expr = expr.replace("ygrey",                   str(ygrey[fs]))
    expr = expr.replace("ymax",                    str(ymax[fs]))
    expr = expr.replace("cmax",                    str(cmax[fs]))
    expr = expr.replace("ymin",                    str(ymin[fs]))
    expr = expr.replace("cmin",                    str(cmin[fs]))
    expr = expr.replace("range_min",               str(range_min[fs]))
    expr = expr.replace("range_half",              str(range_half[fs]))
    expr = expr.replace("range_max",               str(range_max[fs]))
    expr = expr.replace("range_size",              str(range_size[fs]))
    return expr


# feisty2's ChromaReconstructor_faster v3.0 HBD mod by DogWay
def ChromaReconstructor(clip: vs.VideoNode, gpuid: int = 0):
    fulls = GetColorRange(clip) == 0
    w = clip.width
    h = clip.height
    Y = core.std.ShufflePlanes(clip, [0], vs.GRAY)
    Uor = core.std.ShufflePlanes(clip, [1], vs.GRAY)
    Vor = core.std.ShufflePlanes(clip, [2], vs.GRAY)
    device = dict(device_type="auto" if gpuid >= 0 else "cpu", device_id=max(0, gpuid))
    nparams = dict(nns=1, qual=1, etype=1, nsize=0, fulls=fulls, fulld=fulls, \
        target_width=w*2, target_height=h*2, kernel="Bicubic", a1=0.0, a2=0.75, \
        mode="nnedi3cl" if gpuid >= 0 else "znedi3", device=max(0, gpuid))

    ref     = Y.knlm.KNLMeansCL(0, 16, 0, pow(1.464968620512209618455732713658, 6.4), wref=1, **device)
    Luma    = nnedi3.nnedi3_resample(ref, **nparams)
    Uu      = nnedi3.nnedi3_resample(Uor, **nparams)
    Vu      = nnedi3.nnedi3_resample(Vor, **nparams)
    Unew    = Uu.knlm.KNLMeansCL(0, 16, 0, 6.4, wref=0, rclip=Luma, **device).fmtc.resample(w, h, kernel="bicubic", a1=-0.5, a2=0.25)
    Vnew    = Vu.knlm.KNLMeansCL(0, 16, 0, 6.4, wref=0, rclip=Luma, **device).fmtc.resample(w, h, kernel="bicubic", a1=-0.5, a2=0.25)
    U       = core.std.MergeDiff(Unew, core.std.MakeDiff(Unew.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]), Uu.fmtc.resample(w, h, kernel="bicubic", a1=-0.5, a2=0.25)))
    V       = core.std.MergeDiff(Vnew, core.std.MakeDiff(Vnew.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]), Vu.fmtc.resample(w, h, kernel="bicubic", a1=-0.5, a2=0.25)))
    return core.std.ShufflePlanes([Y, U, V], [0, 0, 0], vs.YUV)
