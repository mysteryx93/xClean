/*
xClean 3-pass denoiser
beta 10 (2022-05-15) by Etienne Charland
Supported formats: YUV, RGB, Y
Requires: MaskTools2, MvTools2, KNLMeansCL, BM3DCUDA, aWarpSharpMT, Median, fmtconv, neo_f3kdb, neo_tmedian, nnedi3, nnedi3_resize16, SysInfo, Dogway Extools/TransformsPack

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

All processing is done in YUV444 format. When conv=True, processing is done in YCoCgR, and in OPP colorspace for BM3D.


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

The following chroma upsampling values are also supported (may require additional dependencies)
deep / FCBI / SuperResXBR / Krig / FSRCC / DPID / SSIM / SSIM2 / Nearest / Box / Bilinear / Quadratic / Gauss / Wiener / Spline / Spline16 / Spline36 / Spline64 / 
Spline100 / Spline144 / Spline196 / Spline256 / Jinc / Jinc16 / Jinc36 / Jinc64 / Jinc100 / Jinc144 / Jinc196 / Jinc256 / EWASharp / EWASharp2 / EWASharp4 / 
EWASharper / EWASharper2 / EWASharper4 / EWASharpest / EWASoft / Sinc / SincLin / SinPow / Welch / Cosine / Bessel / Wiener / Hamming / Hann / Kaiser / 
Blackman / Black-Harris / Black-Nuttall / Nuttall / Bohman / Parzen / Lanczos / Ginseng / Flat-Top / MinSide / SoftCubic100 / Robidoux Soft / SoftCubic75 / BilcubicD / 
BilcubicU / Hermite / Robidoux / Centroid / Mitchell-Netravali / Robidoux Sharp / SoftCubic50 / CatMule-Dog / Cub-grange / Catmull-Rom / Didee / Zopti / ZoptiN / ZoptiH / 
Zopti720 / Zopti720U / Zopti1080 / Precise / Sharp / Hatch

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
conv = True. Whether to convert to OPP format for BM3D and YCoCgR for everything else. If false, it will process in standard YUV444.
#################################################  */

function xClean(clip clp, string "chroma", float "sharp", float "rn", bool "deband", int "depth", int "strength", float "m1", float "m2", int "m3", int "outbits", \
        int "dmode", int "rgmode", int "thsad", int "d", int "a", float "h", int "gpuid", int "gpucuda", float "sigma", \
        int "block_step", int "bm_range", int "ps_range", int "radius", bool "bm3d_fast", bool "conv", bool "downchroma", bool "fulls")
{
    nw    = clp.width()
    nh    = clp.height()
    rgb   = clp.isRGB()
    isy   = clp.isY()
    bd    = BitsPerComponent(clp)
    rng   = propNumElements (clp, "_ColorRange")  > 0 ? \
            propGetInt      (clp, "_ColorRange") == 0 : rgb

    defH        = Max(nh, nw / 4 * 3) # Resolution calculation for auto blksize settings
    chroma      = Default(chroma, "nnedi3")
    sharp       = Default(sharp, 9.5)
    rn          = Default(rn, 14.0)
    deband      = Default(deband, False)
    depth       = Default(depth, 0)
    strength    = Default(strength, 20)
    m1          = Default(m1, .6)
    m2          = Default(m2, 2)
    m3          = Default(m3, 2)
    outbits     = Default(outbits, bd)
    dmode       = Default(dmode, 0)
    rgmode      = Default(rgmode, 18)
    thsad       = Default(thsad, 400)
    d           = Default(d, 2)
    a           = Default(a, 2)
    h           = Default(h, 1.4)
    gpuid       = Default(gpuid, 0)
    gpucuda     = Default(gpucuda, gpuid)
    sigma       = Default(sigma, 9.0)
    block_step  = Default(block_step, 4)
    bm_range    = Default(bm_range, 16)
    ps_range    = Default(ps_range, 8)
    radius      = Default(radius, 0)
    bm3d_fast   = Default(bm3d_fast, False)
    conv        = Default(conv, True)
    fulls       = Default(fulls, rng)

    Assert(sharp >= 0 && sharp <= 20, "xClean: sharp must be between 0 and 20")
    Assert(rn    >= 0 && rn    <= 20, "xClean: rn (renoise strength) must be between 0 and 20")
    Assert(depth >= 0 && depth <= 5,  "xClean: depth must be between 0 and 5")
    Assert(strength >= -200 && strength <= 20, "xClean: strength must be between -200 and 20")
    Assert(m1 >= 0 && m1 <  4, "xClean: m1 (MVTools pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    Assert(m2 >= 0 && m2 <  4, "xClean: m2 (BM3D pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    Assert(m3 >= 0 && m3 <= 3, "xClean: m3 (KNLMeansCL pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit)")
    Assert(m1 > 0 || m2 > 0 || m3 > 0, "xClean: At least one pass must be enabled")
    Assert(min(16,max(8,outbits))%2 == 0 || outbits == 32, "xClean: outbits must be 8, 10, 12, 14, 16 or 32")
    #Assert(chroma=="none" || chroma=="bicubic" || chroma=="nnedi3" || chroma=="reconstructor", "xClean: chroma must be none, bicubic, nnedi3 or reconstructor")

    uv = clp
    clp = chroma == "none" ? clp.ExtractY() : clp
    samp = x_ClipSampling(clp)
    matrix = x_GetMatrix(clp)
    cplace = Select(x_GetChromaLoc(clp), "left", "center", "top_left", "left", "left", "left")

    if (isy)
    {
        chroma = "none"
        conv = False
    }
    dochroma = chroma != "none" || rgb
    downchroma = Default(downchroma, chroma != "reconstructor")

    # Reference clips are in RGB or GRAY format, to allow converting to desired formats
    cconv = bd < 16 ? ConvertBits(clp, 16, fulls=fulls) : clp
    cconv = clp.Is444 || rgb || isy ? cconv : \
        chroma == "reconstructor" ? x_ChromaReconstructor(cconv, gpuid=gpuid) : \
        chroma == "bicubic" ? fmtc_resample(clp, csp=bd < 32 ? "YUV444P16" : "YUV444PS" , kernel="bicubic", a1=0, a2=.5, fulls=fulls, fulld=fulls, cplace=cplace) : \
        ConvertFormat(cconv, fmt_out="444", kernel_c=chroma, tv_in=!fulls, cplace_in=cplace)
    cconv = conv && (clp.IsYUV() || clp.IsYUVA()) ? x_ConvertMatrix(cconv, "RGB", fulls) : cconv
    c32   = ConvertBits(cconv, 32, dither=-1, fulls=fulls)
    c16   = ConvertBits(cconv, 16, dither= 0, fulls=fulls)
    c8    = ConvertBits(cconv, 8,  dither= 0, fulls=fulls)
    output = Undefined()

    # Apply MVTools
    if (m1 > 0)
    {
        m1r = m1 == int(m1) ? 1 : m1 - int(m1) # Decimal point is resize factor
        m1 = int(m1)
        c1 = m1 == 3 ? c32 : m1 == 2 ? c16 : c8
        c1 = m1r < 1 ? c1.bicubicResize(nmod(nw * m1r,4), nmod(nh * m1r,4), 0, .75) : c1
        c1 = conv ? RGB_to_YCoCgR(c1) : c1
        output = x_MvTools(c1, defH, thsad)
        sharp1 = max(0, min(20, sharp + (1 - m1r) * .35))
        output = x_PostProcessing(output, c1, defH, strength, sharp1, rn, rgmode, 0)
        # output in YCoCgR format
    }

    # Apply BM3D
    if (m2 > 0)
    {
        m2r = m2 == int(m2) ? 1 :  m2 - int(m2) # Decimal point is resize factor
        m2 = int(m2)
        m2o = max(2, max(m2, m3))
        c2 = m2o==3 ? c32 : c16
        ref = Defined(output) && conv ? RGB_to_OPP(YCoCgR_to_RGB(output)) : output
        ref = isy          ? ref.ExtractY() : ref.ConverttoYUV444()
        ref = Defined(ref) ? ref.spline36resize(nmod(nw * m2r,4), nmod(nh * m2r,4))        : ref
        c2r = m2r < 1      ? c2. bicubicResize (nmod(nw * m2r,4), nmod(nh * m2r,4), 0, .5) : c2
        c2r = ConvertBits(conv ? RGB_to_OPP(c2r) : c2r, 32, dither=-1, fulls=fulls)

        output = x_BM3D(c2r, ref.ConvertBits(32), sigma, gpucuda, block_step, bm_range, ps_range, radius, bm3d_fast)

        output = ConvertBits(output, c2.BitsPerComponent(), dither=-1, fulls=fulls)
        output = conv ? RGB_to_YCoCgR(OPP_to_RGB(output)) : output
        c2     = conv ? RGB_to_YCoCgR(c2)                 : c2
        output = m2r < 1 ? output.spline36resize(nw, nh) : output
        sharp2 = max(0, min(20, sharp + (1 - m2r) * .95))
        output = x_PostProcessing(output, c2, defH, strength, sharp2, rn, rgmode, 1)
        # output in YCoCgR format
    }

    if (Defined(output) && output.height() < nh)
    {
        output = output.spline36resize(nw, nh)
    }

    # Apply KNLMeans
    if (m3 > 0)
    {
        m3 = min(2, m3) # KNL internally computes in 16-bit
        c3 = m3==3 ? c32 : c16
        c3 = conv ? RGB_to_YCoCgR(c3) : c3
        ref = Defined(output) ? ConvertBits(output, c3.BitsPerComponent(), dither=-1, fulls=fulls) : output
        output = x_KnlMeans(c3, ref, d, a, h, gpuid)
        # Adjust sharp based on h parameter.
        sharp3 = max(0, min(20, sharp - .5 + (h/2.8)))
        output = x_PostProcessing(output, c3, defH, strength, sharp3, rn, rgmode, 2)
        # output in YCoCgR format
    }

    # Add Depth (thicken lines for anime)
    if (depth > 0)
    {
        depth2 = -depth*3
        depth  =  depth*2
        output = ex_MakeAddDiff(awarpsharp2(output, depth=depth2, blur=3), awarpsharp2(output, depth=depth, blur=2), output)
    }

    # Apply deband
    if (deband)
    {
        output = output.BitsPerComponent() > 16 ? ConvertBits(output, 16, dither=-1, fulls=fulls) : output
        output = neo_f3kdb(range=16, preset=chroma ? "high" : "luma", grain = defH/15, grainc= chroma ? defH/16 : 0)
    }

    # Convert to desired output format and bitrate
    output = conv ? YCoCgR_to_RGB(output) : output
    if ((clp.IsYUV() || clp.IsYUVA()) && !isy)
    {
        output = x_ConvertMatrix(output, "YUV", fulls, matrix)
        if (downchroma && !clp.Is444())
        {
            output = output.fmtc_resample(css=samp, cplace=cplace, fulls=fulls, fulld=fulls, kernel="bicubic", a1=0, a2=0.5)
        }
    }
    if (output.BitsPerComponent() != outbits)
    {
        output = output.fmtc_bitdepth(bits=outbits, fulls=fulls, fulld=fulls, dmode=dmode)
    }

    # Merge source chroma planes if not processing chroma.
    if (!dochroma && (uv.IsYUV() || uv.IsYUVA()))
    {
        uv = uv.BitsPerComponent() != outbits ? ConvertBits(uv, outbits, dither=0, fulls=fulls) : uv
        output = MergeLuma(uv, output)
    }

    return output
}


function x_PostProcessing(clip clean, clip "c", int "defH", int "strength", float "sharp", float "rn", int "rgmode", int "method")
{
    bi    = clean.BitsPerComponent()
    bic   = c.    BitsPerComponent()
    fulls = propNumElements(clean, "_ColorRange")  > 0 ? \
            propGetInt     (clean, "_ColorRange") == 0 : isRGB(clean)

    if (rgmode == 0)
    {
        sharp = 0
        rn = 0
    }

    # Run at least in 16-bit
    clean = bi  < 16 ? ConvertBits(clean, 16, dither=-1, fulls=fulls) : clean
    c     = bic < 16 ? ConvertBits(c,     16, dither=-1, fulls=fulls) : c

    # Separate luma and chroma
    filt  = clean
    clean = clean.ExtractY()
    cy    = c.ExtractY()

    # Spatial luma denoising
    clean2 = rgmode > 0 ? clean.RemoveGrain(rgmode) : clean

    # Apply dynamic noise reduction strength based on Luma
    if (strength <= 0)
    {
        # Slightly widen the exclusion mask to preserve details and edges
        cleanm = cy.ex_Expand(round(0.0014*defH+0.786))

        # Adjust mask levels
        cleanm = cleanm.Levels((fulls ? 0 : 16) - strength, fulls ? 255 : 235, 0.85, 0, 255+strength)

        # Merge based on luma mask
        clean  = ex_Merge(clean,  cy, cleanm)
        clean2 = ex_Merge(clean2, cy, cleanm)
        filt   = ex_Merge(filt,   c,  cleanm)
    }
    else if (strength < 20)
    {
        # Reduce strength by partially merging back with original
        str    = 0.2+0.04*strength
        clean  = Merge(cy, clean,  str)
        clean2 = Merge(cy, clean2, str)
        filt   = Merge(c, filt,    str)
    }

    # Unsharp filter for spatial detail enhancement
    if (sharp > 0)
    {
        mult    = method == 2 ? .69 : method == 1 ? .14 : 1
        sharp   = min(50, (15 + defH * sharp * 0.0007) * mult)
        clsharp = ex_MakeDiff(clean, clean2.ex_blur(0.08+0.03*sharp))
        clsharp = ex_AddDiff(clean2, Repair(clsharp.neo_tmedian(), clsharp, 12))
    }

    # If selected, combining ReNoise
    noise_diff = ex_MakeDiff(clean2, cy)

    if (rn > 0)
    {
        i    = ex_bs(1, 8, bi, fulld=false)
        mul  = 1.008+0.00016*rn
        opa  = 0.3+rn*0.035
        clean2 = ex_lutxyz(clean2, noise_diff.neo_tmedian(), clean, \
        Format("x dup dup dup dup {mul} y * + range_half - - {opa} * - - z range_max z - min Z@ 32 {i} * < 0 Z 45 {i} * > range_max 0 Z 35 {i} * - range_max 32 {i} * 65 {i} * - / * - ? ? range_max / * -"))
    }

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.ex_Binarize(invert=true)
    clean2 = rgmode > 0 ? ex_Merge(clean2, sharp > 0 ? clsharp : clean, ex_logic(noise_diff, clean.ex_edge("kroon",0,110), "max")) : clean2

    # Combining result of luma and chroma cleaning
    return (c.IsYUV() || c.IsYUVA()) ? MergeLuma(filt, clean2) : clean2
}


# mClean denoising method
function x_MvTools(clip c, int "defH", int "thSAD")
{
    bd = BitsPerComponent  (c)
    fulls = propNumElements(c, "_ColorRange")  > 0 ? \
            propGetInt     (c, "_ColorRange") == 0 : isRGB(c)
    cy = c.ExtractY()

    sc = defH > 2880 ? 8 : defH > 1440 ? 4 : defH > 720 ? 2 : 1
    blksize = defH / sc > 360 ? 16 : 8
    overlap = blksize > 12 ? 6 : 2
    pel = defH > 720 ? 1 : 2
    lambda = int(777 * pow(blksize, 2) / 64)
    truemotion = defH <= 720

    ref = c.removegrain(12)
    super1 = MSuper(ref, hpad=blksize, vpad=blksize, pel=pel, rfilter=3, sharp=2)
    super2 = MSuper(c,   hpad=blksize, vpad=blksize, pel=pel, rfilter=1, levels=1)

    # Analysis
    bvec4       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=4, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec3       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=3, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec2       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=2, badsad=1100, lsad=1120, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec1       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=1, badsad=1500, lsad=980, badrange=27, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec1       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=1, badsad=1500, lsad=980, badrange=27, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec2       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=2, badsad=1100, lsad=1120, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec3       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=3, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec4       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=4, blksize=blksize, overlap=overlap, search=0, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)

    # Applying cleaning
    clean = MDegrain4(c, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)

    if (bd < 16)
    {
        clean = clean.ConvertBits(16, dither=-1, fulls=fulls)
        c     = c.    ConvertBits(16, dither=-1, fulls=fulls)
    }

    if (c.IsYUV() || c.IsYUVA())
    {
        uv = ex_AddDiff(clean, neo_tmedian(ex_MakeDiff(c, clean, Y=1, UV=3)), Y=1, UV=3)
        clean = MergeLuma(uv, clean)
    }
    return clean
}


# BM3D denoising method
function x_BM3D(clip c, clip "ref", float "sigma", int "gpuid", int "block_step", int "bm_range", int "ps_range", int "radius", bool "bm3d_fast")
{
    fulls = propNumElements (c, "_ColorRange")  > 0 ? \
            propGetInt      (c, "_ColorRange") == 0 : isRGB(c)
    matrix = x_GetMatrixStr(c, fulls)
    chroma = (c.IsYUV() || c.IsYUVA()) && !c.IsY()
    clean = gpuid >= 0 ? \
        BM3D_CUDA(c, sigma=sigma, ref=ref, block_step=block_step, bm_range=bm_range, ps_range=ps_range, device_id=gpuid, fast=bm3d_fast, chroma=chroma, radius=radius) : \
        BM3D_CPU (c, sigma=sigma, ref=ref, block_step=block_step, bm_range=bm_range, ps_range=ps_range, chroma=chroma, radius=radius)
    clean = radius > 0 ? clean.BM3D_VAggregate(radius=radius) : clean
    return clean
}


# KnlMeansCL denoising method, useful for dark noisy scenes
function x_KnlMeans(clip c, clip ref, int "d", int "a", float "h", int "gpuid")
{
    bd    = BitsPerComponent(c)
    fulls = propNumElements (c, "_ColorRange")  > 0 ? \
            propGetInt      (c, "_ColorRange") == 0 : isRGB(c)
    ref = Defined(ref) ? ref.BitsPerComponent == c.BitsPerComponent ? ref : ref.ConvertBits(bd, dither=0, fulls=fulls) : Undefined()
    src = c
    device = gpuid >= 0 ? "auto" : "cpu"
    gpuid = max(0, gpuid)
    if (c.IsY)
    {
        output = c.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, device_type=device, device_id=gpuid)
    }
    else if (c.Is444())
    {
        output = c.KNLMeansCL(d=d, a=a, h=h, channels="YUV", rclip=ref, device_type=device, device_id=gpuid)
    }
    else
    {
        clean = c.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, device_type=device, device_id=gpuid)
        uv = c.KNLMeansCL(d=d, a=a, h=h/2, channels="UV", rclip=ref, device_type=device, device_id=gpuid)
        output = MergeLuma(uv, clean)
    }
    return output
}


# Get frame properties
function x_GetFrameProp(clip c, string "name", int "default")
{
    return propGetType(c, name) > 0 ? propGetInt(c, name) : default
}

function x_GetMatrixStr(clip c, bool "fullrange")
{
    fulls = propNumElements(c, "_ColorRange")  > 0 ? \
            propGetInt     (c, "_ColorRange") == 0 : isRGB(c)
    matrix = x_GetMatrix(c)
    full = Default(fullrange, fulls)
    return matrix == 6 ? full ? "PC.601" : "Rec601" : full ? "PC.709" : "Rec709"
}


function x_GetMatrix(clip c)
{
    if (c.IsRGB)
    {
        return 0
    }
    matrix = x_GetFrameProp(c, "_Matrix", 1)
    return matrix==0 || matrix==2 ? 6 : matrix
}


function x_GetChromaLoc(clip c)
{
    return x_GetFrameProp(c, "_ChromaLocation", 0)
}



# Converts matrix into desired format. If matrix is not specified, it will read matrix from source frame property.
function x_ConvertMatrix(clip c, string "family", bool "fulls", int "matrix")
{
    matrix = Default(matrix, x_GetMatrix(c))
    csp = BuildPixelType(family, c.BitsPerComponent())
    if (matrix == 10)
    {
        return c.fmtc_matrix2020cl(csp=csp, full=fulls)
    }
    else
    {
        mat = Select(matrix, "RGB", "709", "601", "", "FCC", "601", "601", "240", "YCgCo", "2020", "", "", "", "")
        Assert(mat != "", Format("ConvertMatrix: matrix {} is not supported.", matrix))
        return c.fmtc_matrix(csp=csp, mat=mat, fulls=fulls, fulld=fulls)
    }
}

function x_ClipSampling(clip c)
{
    return c.IsY() ? "GRAY" : c.IsRGB() ? "RGB" : \
        c.Is444() ? "444" : c.Is422() ? "422" : c.Is420() ? "420" : "UNKNOWN"
}


# DogWay's ChromaReconstructor_faster
function x_ChromaReconstructor(clip src, int "gpuid", int "radius", float "str", int "threads") {

        w       = src.width()
        h       = src.height()
        bi      = BitsPerComponent(src)
        fs      = propNumElements (src,"_ColorRange")  > 0 ? \
                  propGetInt      (src,"_ColorRange") == 0 : false

        radius  = Default(radius, 16)
        str     = Default(str,   6.4)
        threads = Default(threads, 4)

        cores   = SI_PhysicalCores()
        threads = SI_LogicalCores()

        Y   = ExtractY(src)
        Uor = ExtractU(src)
        Vor = ExtractV(src)

        ref     =   Y.KNLMeansCL (0, radius, 0, pow(1.464968620512209618455732713658, str), "auto", device_id=gpuid, wref=1)
        Luma    = ref.ConvertBits(8,dither=-1,fulls=fs).nnedi3_rpow2(rfactor=2, nns=1, qual=1, etype=1, nsize=0, threads=cores, prefetch=(threads+cores)/2, range=fs?1:2, cshift="spline16resize").ConvertBits(bi,fulls=fs)
        Uu      = Uor.ConvertBits(8,dither=-1,fulls=fs).nnedi3_rpow2(rfactor=2, nns=1, qual=1, etype=1, nsize=0, threads=cores, prefetch=(threads+cores)/2, fwidth=w*2, fheight=h*2, ep0=6, cshift="blackmanresize", mpeg2=false,range=fs?1:2).ConvertBits(bi,fulls=fs)
        Vu      = Vor.ConvertBits(8,dither=-1,fulls=fs).nnedi3_rpow2(rfactor=2, nns=1, qual=1, etype=1, nsize=0, threads=cores, prefetch=(threads+cores)/2, fwidth=w*2, fheight=h*2, ep0=6, cshift="blackmanresize", mpeg2=false,range=fs?1:2).ConvertBits(bi,fulls=fs)
        Unew    = Uu.KNLMeansCL  (0, radius, 0, str, "auto", wref=0, rclip=Luma, device_id=gpuid).BicubicResize(w, h, b=0.0, c=0.5)
        Vnew    = Vu.KNLMeansCL  (0, radius, 0, str, "auto", wref=0, rclip=Luma, device_id=gpuid).BicubicResize(w, h, b=0.0, c=0.5)
        U       = ex_makeadddiff(Unew, Removegrain(Unew, 20), Uu.BicubicResize(w, h, b=0.0, c=0.5))  # Sharpening
        V       = ex_makeadddiff(Vnew, Removegrain(Vnew, 20), Vu.BicubicResize(w, h, b=0.0, c=0.5))  # Sharpening
        return CombinePlanes(Y, U, V, planes="YUV")
}