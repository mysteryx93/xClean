/*
xClean 3-pass denoiser
beta 8 (2021-11-10) by Etienne Charland
Supported formats: YUV, RGB, Y
Requires: MaskTools2, MvTools2, KNLMeansCL, BM3D, aWarpSharp2, Median, BM3DCUDA, fmtconv, neo_f3kdb, nnedi3, nnedi3_resize16

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
#################################################  */

function xClean(clip clip, string "chroma", float "sharp", float "rn", bool "deband", int "depth", int "strength", float "m1", int "m2", int "m3", int "outbits", \
        int "dmode", int "rgmode", int "thsad", int "d", int "a", float "h", int "gpuid", int "gpucuda", float "sigma", \
        int "block_step", int "bm_range", int "ps_range", bool "conv", bool "downchroma", bool "fulls")
{
    current_frame = 0   # Allow reading frame properties of first frame
    width 		= clip.width
    height 		= clip.height
    defH        = Max(height, width / 4 * 3) # Resolution calculation for auto blksize settings
    bd          = clip.BitsPerComponent()
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
    conv		= Default(conv, True)
    fulls   	= Default(fulls, x_GetColorRange(clip) == 0)

    Assert(sharp >= 0 && sharp <= 20, "xClean: sharp must be between 0 and 20")
    Assert(rn >= 0 && rn <= 20, "xClean: rn (renoise strength) must be between 0 and 20")
    Assert(depth >= 0 && depth <= 5, "xClean: depth must be between 0 and 5")
    Assert(strength >= -200 && strength <= 20, "xClean: strength must be between -200 and 20")
    Assert(m1 >= 0 && m1 < 4, "xClean: m1 (MVTools pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    Assert(m2 >= 0 && m2 < 4, "xClean: m2 (BM3D pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit), plus an optional downscale ratio as decimal (eg: 2.6 resizes to 60% in 16-bit)")
    Assert(m3 >= 0 && m3 <= 3, "xClean: m3 (KNLMeansCL pass) can be 0 (disabled), 1 (8-bit), 2 (16-bit) or 3 (32-bit)")
    Assert(m1 > 0 || m2 > 0 || m3 > 0, "xClean: At least one pass must be enabled")
    Assert(outbits == 8 || outbits == 10 || outbits == 12 || outbits == 14 || outbits == 16 || outbits == 32, "xClean: outbits must be 8, 10, 12, 14, 16 or 32")
    Assert(chroma=="none" || chroma=="bicubic" || chroma=="nnedi3" || chroma=="reconstructor", "xClean: chroma must be none, bicubic, nnedi3 or reconstructor")

    uv = clip
	clip = chroma == "none" ? clip.ExtractY() : clip
    samp = ClipSampling(clip)
    matrix = GetMatrix(clip)
    cplace = Select(GetChromaLoc(clip), "left", "center", "top_left", "left", "left", "left")

    if (clip.IsY)
    {
        chroma = "none"
        conv = False
    }
    dochroma = chroma != "none" || IsRGB(clip)
    downchroma = Default(downchroma, chroma != "reconstructor")
    
    # Reference clips are in RGB or GRAY format, to allow converting to desired formats
    cconv = clip.IsYV24 || clip.ISRGB || clip.IsY ? clip : \
        chroma == "reconstructor" ? ChromaReconstructor(clip, gpuid) : \
        chroma == "nnedi3" ? nnedi3_resize16(clip, tv_range=!fulls, output="YV24") : \
        fmtc_resample(clip, csp=bd < 32 ? "YUV444P16" : "YUV444PS" , kernel="bicubic", a1=0, a2=.5)
    cconv = conv && (cilp.IsYUV || clip.IsYUVA) ? cchroma.ConvertToRGB(matrix_in=GetMatrix(clip)) : clip
    c32 = x_ConvertBits(cconv, 32, fulls, False)
    c16 = x_ConvertBits(cconv, 16, fulls, True)
    c8 = x_ConvertBits(cconv, 8, fulls, True)
    output = Undefined()

    # Apply MVTools
    if (m1 > 0)
    {
        m1r = m1 == int(m1) ? 1 : m1 - int(m1) # Decimal point is resize factor
        m1 = int(m1)
        c1 = m1 == 3 ? c32 : m1 == 2 : c16 : c8
        c1 = m1r < 1 ? c1.fmtc_resample(int(width * m1r)/4*4, int(height * m1r)/4*4, kernel="bicubic", a1=0, a2=.75) : c1
        c1 = conv ? RGB_to_YCgCoR(c1, fulls) : c1
        output = MvTools(c1, defH, thsad)
        sharp1 = max(0, min(20, sharp + (1 - m1r) * .35))
        output = PostProcessing(output, c1, defH, strength, sharp1, rn, rgmode, 0)
        # output in YCgCoR format
	}

    # Apply BM3D
    if (m2 > 0)
    {
        m2r = m2 == int(m2) ? 1 :  m2 - int(m2) # Decimal point is resize factor
        m2 = int(m2)
        m2o = max(2, max(m2, m3))
        c2 = m2o==3 ? c32 : c16
        ref = output && conv ? RGB_to_OPP(YCgCoR_to_RGB(output, fulls), fulls) : output
        ref = Defined(ref) ? x_ConvertBits(ref.fmtc_resample(int(width * m2r)//4*4, int(height * m2r)//4*4, kernel="spline36"), 32, fulls, False) : ref
        c2r = m2r < 1 ? c2.fmtc_resample(int(width * m2r)//4*4, int(height * m2r)//4*4, kernel="bicubic", a1=0, a2=.5) : c2
        c2r = x_ConvertBits(conv ? RGB_to_OPP(c2r, fulls) : c2r, 32, fulls, False)

        output = BM3D(c2r, ref, sigma, gpucuda, block_step, bm_range, ps_range, radius, bm3d_fast)
        
        output = x_ConvertBits(output, c2.BitsPerComponent, fulls, False)
        output = conv ? RGB_to_YCgCoR(OPP_to_RGB(output, fulls), fulls) : output
        c2 = conv ? RGB_to_YCgCoR(c2, fulls) : c2
        output = m2r < 1 ? output.fmtc_resample(width, height, kernel="spline36") : output
        sharp2 = max(0, min(20, sharp + (1 - m2r) * .95))
        output = PostProcessing(output, c2, defH, strength, sharp2, rn, rgmode, 1)
        # output in YCgCoR format
	}
	
    if (Defined(output) and output.height < height)
    {
        output = output.fmtc_resample(width, height, kernel="spline36")
	}

    # Apply KNLMeans
    if (m3 > 0)
    {
        m3 = min(2, m3) # KNL internally computes in 16-bit
        c3 = m3==3 ? c32 : c16
        c3 = conv ? RGB_to_YCgCoR(c3, fulls) : c3
        ref = x_ConvertBits(output, c3.BitsPerComponent, fulls, False) if Defined(output) else Undefined()
        output = KnlMeans(c3, ref, d, a, h, gpuid)
        # Adjust sharp based on h parameter.
        sharp3 = max(0, min(20, sharp - .5 + (h/2.8)))
        output = PostProcessing(output, c3, defH, strength, sharp3, rn, rgmode, 2)
        # output in YCgCoR format
    }    

    # Add Depth (thicken lines for anime)
    if (depth > 0)
    {
		depth2 = -depth*3
		depth = depth*2
		output = mt_adddiff(output, mt_makediff(awarpsharp2(output, depth=depth2, blur=3), awarpsharp2(output, depth=depth, blur=2)))
    }
    
    # Apply deband
    if (deband)
    {
        output = output.BitsPerComponent > 16 ? x_ConvertBits(output, 16, fulls, False) : output
		output = neo_f3kdb(range=16, preset=chroma ? "high" : "luma", grain = defH/15, grainc= chroma ? defH/16 : 0)
    }

    # Convert to desired output format and bitrate
    output = conv ? YCgCoR_to_RGB(output, fulls) : output
    if (clip.IsYUV || clip.IsYUVA)
    {
        output = ConvertMatrix(output, vs.YUV, fulls, matrix)
        if (downchroma && !clip.IsYV24)
        {
            output = output.fmtc_resample(css=samp, cplace=cplace, fulls=fulls, fulld=fulls, kernel="bicubic", a1=0, a2=0.5)
        }
    }
    if (output.bits_per_component != outbits)
    {
        output = output.fmtc_bitdepth(bits=outbits, fulls=fulls, fulld=fulls, dmode=dmode)
    }
     
    # Merge source chroma planes if not processing chroma.
    if (!dochroma && (uv.IsYUV || uv.IsYUVA)
    {
        uv = uv.bits_per_component != outbits ? x_ConvertBits(uv, outbits, fulls, True) : uv
        output = MergeChroma(output, uv)
    }
    
    return output
}


function x_PostProcessing(clip clean, clip "c", int "defH", int "strength", float "sharp", float "rn", int "rgmode", int "method", bool "fullrange")
{
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
}


# mClean denoising method
function x_MvTools(clip c, bool "chroma", int "defH", int "thSAD")
{
    bd = c.BitsPerComponent
    cy = c.ExtractY()

    sc = defH > 2880 ? 8 : defH > 1440 ? 4 : defH > 720 ? 2 : 1
    blksize = defH / sc > 360 ? 16 : 8
    overlap = blksize > 12 ? 6 : 2
    pel = defH > 720 ? 1 : 2
    lampa = int(777 * pow(blksize, 2) / 64)
    truemotion = defH <= 720

    ref = c.Blur(1.0)
    super1 = S(chroma ? ref : ref.ExtractY(), hpad=blksize, vpad=blksize, pel=pel, rfilter=4, sharp=1)
    super2 = S(chroma ? c : cy, hpad=blksize, vpad=blksize, pel=pel, rfilter=1, levels=1)

    # Analysis
    bvec4       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=4, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec3       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=3, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec2       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=2, badsad=1100, lsad=1120, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    bvec1       =  MRecalculate(super1, MAnalyse(super1, isb=true, delta=1, badsad=1500, lsad=980, badrange=27, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec1       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=1, badsad=1500, lsad=980, badrange=27, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec2       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=2, badsad=1100, lsad=1120, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec3       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=3, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)
    fvec4       =  MRecalculate(super1, MAnalyse(super1, isb=false, delta=4, blksize=blksize, overlap=overlap, search=5, truemotion=truemotion),
                \  blksize=blksize, overlap=overlap, search=5, truemotion=truemotion, lambda=lambda, thSAD=180)

    # Applying cleaning
    clean = MDegrain4(chroma ? c : cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)

    clean = bd < 16 ? clean.fmtc_bitdepth(bits=16) : clean
    c = bd < 16 ? c.fmtc_bitdepth(bits=16) : c

    uv = chroma ? mt_adddiff(clean, neo_tmedian(mt_makediff(c, clean, Y=1, U=3, V=3)), Y=1, U=3, V=3) : c
    clean = c.IsY ? clean : MergeLuma(uv, clean, 1)
    return clean
}


# SpotLess denoising method (m1=4) EXPERIMENTAL
Function x_SpotLess(clip c, int "RadT", int "ThSAD", int "ThSAD2", int "pel", bool "chroma", int "BlkSz", int "Olap", bool "tm", bool "glob", float "bBlur")
{
    myName = "SpotLess: "
    RadT     = Default(RadT,1)        # Temporal radius. (MCompensate arg)
    ThSAD    = Default(ThSAD,10000)   # SAD threshold at radius 1 (Default Nearly OFF).
    ThSAD2   = Default(ThSAD2,ThSAD)  # SAD threshold at radius RadT.
    Pel      = Default(pel,2)         # Default 2. 1, 2, or 4. Maybe set 1 for HD+. (1=precision to pixel, 2=precision to half pixel, 4=quarter pixel)
    Chroma   = Default(chroma,True)   # MAnalyse chroma arg. If set to true, use chroma in block matching.
    BlkSz    = Default(BlkSz,8)       # Default 8. MAnalyse BlkSize. Bigger blksz quicker and perhaps  better, esp for HD clips. Maybe also better where BIG noise.
    OLap     = Default(OLap, BlkSz/2) # Default half of BlkSz.
    Tm       = Default(tm,True)       # TrueMotion, Some folk swear MAnalyse(truemotion=false) is better.
    Glob     = Default(glob,True)     # Default True, Allow set MAnalyse(global) independently of TrueMotion.
    Bblur    = Default(bblur,0.0)     # Default OFF
    Assert(1 <= RadT,myName + " 1 <= RadT")
    Assert(0.0 <= bblur <= 1.58, myName + "0.0 <= bblur <= 1.58")
    Assert(pel==1 || pel==2 || pel==4, myName + "pel==1 || pel==2 || pel==4")
    pad      = max(BlkSz,8)
    sup      = (bBlur<=0.0  ? c   : c.blur(bblur)).MSuper(hpad=pad,vpad=pad,pel=pel,sharp=2)
    sup_rend = (bBlur<=0.0) ? sup : c.MSuper(hpad=pad,vpad=pad,pel=pel,sharp=2,levels=1) # Only 1 Level required where not MAnalyse-ing.
    MultiVec = sup.MAnalyse(multi=true, delta=RadT,blksize=BlkSz,overlap=OLap,chroma=Chroma,truemotion=Tm,global=Glob)
    c.MCompensate(sup_rend, MultiVec, tr=RadT, thSad=ThSAD, thSad2=ThSAD2)
    MedianBlurTemporal(radiusY=0,radiusU=0,radiusV=0,temporalradius=RadT)  # Temporal median blur only [not spatial]
    SelectEvery(RadT*2+1,RadT)                                             # Return middle frame
}


# BM3D denoising method
function x_BM3D(clip c, float "sigma", int "gpuid", bool  "chroma", clip "ref", int "m", int "block_step", int "bm_range", int "ps_range", bool "fullrange")
{
    matrix = x_GetMatrixStr(c, fullrange)
    clean = chroma ? c.fmtc_bitdepth(bits=32).ConvertToPlanarRGB(matrix=matrix) : c.ExtractY().fmtc_bitdepth(bits=32)
    ref = Defined(ref) ? chroma ? ref.fmtc_bitdepth(bits=32).ConvertToPlanarRGB(matrix=matrix, ChromaInPlacement=loc) : ref.ExtractY().fmtc_bitdepth(bits=32) : Undefined()
    clean = gpuid >= 0 ? \
        BM3D_CUDA(clean, sigma=sigma, ref=ref, block_step=block_step, bm_range=bm_range, ps_range=ps_range, device_id=gpuid, fast=False) : \
        BM3D_CPU (clean, sigma=sigma, ref=ref, block_step=block_step, bm_range=bm_range, ps_range=ps_range)
    return x_ConvertToM(clean, c, m, fullrange)
}


# KnlMeansCL denoising method, useful for dark noisy scenes
function x_KnlMeans(clip c, int "d", int "a", float "h", int "gpuid", bool "chroma", clip "ref")
{
    ref = Defined(ref) ? ref.BitsPerComponent == c.BitsPerComponent ? ref : ref.fmtc_bitdepth(bits=c.BitsPerComponent) : Undefined()
    device = gpuid >= 0 ? "auto" : "cpu"
    gpuid = max(0, gpuid)
    output = c.IsY ? c.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, device_type=device, device_id=gpuid) : Undefined()
    output = c.Is444 ? c.KNLMeansCL(d=d, a=a, h=h, channels="YUV", rclip=ref, device_type=device, device_id=gpuid) : output
    clean = Defined(output) ? output : c.KNLMeansCL(d=d, a=a, h=h, channels="Y", rclip=ref, device_type=device, device_id=gpuid)
    uv = Defined(output) || !chroma ? clean : c.KNLMeansCL(d=d, a=a, h=h/2, channels="UV", rclip=ref, device_type=device, device_id=gpuid)
    return MergeLuma(uv, clean, 1)
}


# Get frame properties
function x_GetFrameProp(clip c, string "name", int "default")
{
    return propGetType(c, name) > 0 ? propGetInt(c, name) : default
}

function x_GetMatrixStr(clip c, bool "fullrange")
{
    matrix = x_GetMatrix(c)
    full = Default(fullrange, x_GetColorRange(c) == 0)
    return matrix == 6 ? full ? "PC.601" : "Rec601" : full ? "PC.709" : "Rec709"
}

function x_GetColorRange(clip c)
{
    return x_GetFrameProp(c, "_ColorRange", 1)
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

function x_GetMatrixStr(clip c)
{	
	fulls = x_GetColorRange(c) == 0
	matrix = x_GetMatrix(c)
	
}

#function x_GetChromaLoc(clip c)
#{
#    return x_GetFrameProp(c, "_ChromaLocation", 0)
#}

#function x_GetChromaLocStr(clip c)
#{
#    loc = x_GetChromaLoc(c)
#    return loc == 1 ? "MPEG1" : "MPEG2"
#}

function x_ConvertBits(clip c, int "bits", bool "fulls", bool "dither")
{
	fulls = Default(fulls, c.IsRGB)
	dither = Default(dither, False)
	return c.ConvertBits(bits=bits, fulls=fulls, fulld=fulls, dither = dither ? 0 : -1)
}

# HBD constants 3D look up table
#
# * YUV and RGB mid-grey is 127.5 (rounded to 128) for PC range levels,
#   this translates to a value of 125.5 in TV range levels. Chroma is always centered, so 128 regardless.
function ex_dlut(string "str", int "bits", bool "fulls")
{
    str  = Default(str, "")
    bits = Default(bits, 8)
    fs   = Default(fulls, false)

    bitd =
\     (bits ==  8         ) ? 0
\   : (bits == 10         ) ? 1
\   : (bits == 12         ) ? 2
\   : (bits == 14         ) ? 3
\   : (bits == 16         ) ? 4
\   : (bits == 24         ) ? 5
\   : (bits == 32         ) ? 6
\   :  Assert (false, string(bits)+": Unsupported bit depth.")


    #                            8-bit UINT      10-bit UINT          12-bit UINT          14-bit UINT            16-bit UINT         24-bit UINT               32-bit Ufloat
    range_min   = Select (bitd,  [  0.,  0.],    [   0.,   0.   ],    [   0.,   0.   ],    [    0.,    0.   ],    [    0.,    0.],    [       0.,       0.],    [       0.,       0.])
    ymin        = Select (bitd,  [ 16., 16.],    [  64.,  64.   ],    [ 256., 257.   ],    [ 1024., 1028.   ],    [ 4096., 4112.],    [ 1048576., 1052672.],    [  16/255.,  16/255.])
    cmin        = Select (bitd,  [ 16., 16.],    [  64.,  64.   ],    [ 256., 257.   ],    [ 1024., 1028.   ],    [ 4096., 4112.],    [ 1048576., 1052672.],    [  16/255.,  16/255.])
    ygrey       = Select (bitd,  [126.,126.],    [ 502., 504.   ],    [2008.,2016.   ],    [ 8032., 8063.   ],    [32128.,32254.],    [ 8224768., 8256896.],    [ 125.5/255.,125.5/255.])
    range_half  = Select (bitd,  [128.,128.],    [ 512., 514.   ],    [2048.,2056.   ],    [ 8192., 8224.   ],    [32768.,32896.],    [ 8388608., 8421376.],    [ 128/255., 128/255.])
    yrange      = Select (bitd,  [219.,219.],    [ 876., 879.   ],    [3504.,3517.688],    [14016.,14070.750],    [56064.,56283.],    [14352384.,14408448.],    [ 219/255., 219/255.])
    crange      = Select (bitd,  [224.,224.],    [ 896., 899.500],    [3584.,3598.   ],    [14336.,14392.   ],    [57344.,57568.],    [14680064.,14737408.],    [ 224/255., 224/255.])
    ymax        = Select (bitd,  [235.,235.],    [ 940., 943.672],    [3760.,3774.688],    [15040.,15098.750],    [60160.,60395.],    [15400960.,15461120.],    [ 235/255., 235/255.])
    cmax        = Select (bitd,  [240.,240.],    [ 960., 963.750],    [3840.,3855.   ],    [15360.,15420.   ],    [61440.,61680.],    [15728640.,15790080.],    [ 240/255., 240/255.])
    range_max   = Select (bitd,  [255.,255.],    [1020.,1023.984],    [4080.,4095.938],    [16320.,16383.750],    [65280.,65535.],    [16711680.,16776960.],    [       1.,       1.])
    range_size  = Select (bitd,  [256.,256.],    [1024.,1024.   ],    [4096.,4096.   ],    [16384.,16384.   ],    [65536.,65536.],    [16777216.,16777216.],    [       1.,       1.])

    fs  = fs ? 1 : 0
    str = ReplaceStr(str, "ymax ymin - range_max /", string(yrange[fs]/range_max[fs]))
    str = ReplaceStr(str, "cmax cmin - range_max /", string(crange[fs]/range_max[fs]))
    str = ReplaceStr(str, "cmax ymin - range_max /", string(crange[fs]/range_max[fs]))
    str = ReplaceStr(str, "range_max ymax ymin - /", string(range_max[fs]/yrange[fs]))
    str = ReplaceStr(str, "range_max cmax cmin - /", string(range_max[fs]/crange[fs]))
    str = ReplaceStr(str, "range_max cmax ymin - /", string(range_max[fs]/crange[fs]))
    str = ReplaceStr(str, "ymax ymin -",             string(yrange[fs]))
    str = ReplaceStr(str, "cmax ymin -",             string(crange[fs]))
    str = ReplaceStr(str, "cmax cmin -",             string(crange[fs]))

    str = ReplaceStr(str, "ygrey",                   string(ygrey[fs]))
    str = ReplaceStr(str, "ymax",                    string(ymax[fs]))
    str = ReplaceStr(str, "cmax",                    string(cmax[fs]))
    str = ReplaceStr(str, "ymin",                    string(ymin[fs]))
    str = ReplaceStr(str, "cmin",                    string(cmin[fs]))
    str = ReplaceStr(str, "range_min",               string(range_min[fs]))
    str = ReplaceStr(str, "range_half",              string(range_half[fs]))
    str = ReplaceStr(str, "range_max",               string(range_max[fs]))
    str = ReplaceStr(str, "range_size",              string(range_size[fs]))

    return str
}