#!/usr/bin/env python

############
# ppspline #
############

# ppspline is a command-line program to make a frequency-parameterized model of
#    wideband profile evolution.  The parameterization is good within the range
#    of data frequencies, provided there are not huge gaps in frequency.  For
#    an input nchan x nbin average portrait of aligned profiles, the model is a
#    B-spline representation of the curve traced out by the nchan profile
#    amplitude vectors in an nbin vector space.  Since the profile shapes are
#    highly correlated, the B-spline representation can be reduced to << nbin
#    dimensions, and is limited to ten dimensions.  Therefore, the profile
#    variability is decomposed using PCA and a small number of eigenprofiles
#    that encompass most of the profile evolution are selected.  The input data
#    should be high S/N, averaged, and "aligned" (e.g. output from
#    ppalign.py).  Pre-normalization of the input is encouraged (specifically
#    using normalize_portrait('prof')), as is using smooth=True in the
#    make_model function.

# Written by Timothy T. Pennucci (TTP; tim.pennucci@nanograv.org).

from __future__ import division
from __future__ import print_function

from past.utils import old_div
from pplib import *
from pplib import _smart_smooth_profile, _default_try_nlevels


def _fit_pca_and_smooth(port, freqs, pca_weights, max_ncomp, smooth,
                        snr_cutoff, rchi2_tol, wavelet, quiet=False, **kwargs):
    """
    Do PCA on port and (optionally) wavelet-smooth the mean profile and the
        significant eigenprofiles.

    Returns (mean_prof, eigval, eigvec, ieig, ncomp, smooth_mean_prof,
        smooth_eigvec); smooth_mean_prof and smooth_eigvec are None if
        smooth is False.

    port is an nchan x nbin array of data values.
    freqs is an nchan array of the frequencies corresponding to port; it is
        unused here but accepted for a uniform call signature with
        _fit_spline_curve(...).
    pca_weights are the nchan weights used in pca(...).
    max_ncomp, smooth, snr_cutoff, rchi2_tol, and **kwargs are as in
        make_spline_model(...); smooth=True requires pywt.
    wavelet is the name of the mother wavelet passed to smart_smooth(...)
        (and on to wavelet_smooth(...)) for both the mean profile and the
        eigenprofiles.
    quiet=True suppresses output.
    """
    mean_prof = old_div((port.T * pca_weights).T.sum(axis=0), pca_weights.sum())
    eigval, eigvec = pca(port, mean_prof, pca_weights, quiet=quiet)
    if max_ncomp is None:
        return_max = 10
    else:
        return_max = min(max_ncomp, 10)
    smooth_mean_prof = smooth_eigvec = None
    if smooth:
        if 'pywt' not in sys.modules:
            raise ImportError("You failed to import pywt and need PyWavelets to use smooth=True!")
        ieig, smooth_eigvec = find_significant_eigvec(eigvec, check_max=10,
                                                       return_max=return_max, snr_cutoff=snr_cutoff,
                                                       return_smooth=True, rchi2_tol=rchi2_tol,
                                                       wavelet=wavelet, **kwargs)
        smooth_mean_prof = smart_smooth(mean_prof, rchi2_tol=rchi2_tol,
                                        wavelet=wavelet)
    else:
        ieig = find_significant_eigvec(eigvec, check_max=10,
                                       return_max=return_max, snr_cutoff=snr_cutoff,
                                       return_smooth=False, rchi2_tol=rchi2_tol,
                                       wavelet=wavelet, **kwargs)
    ncomp = len(ieig)
    return (mean_prof, eigval, eigvec, ieig, ncomp, smooth_mean_prof,
            smooth_eigvec)


def _fit_spline_curve(proj_port, freqs, spl_weights, noise_stds, k, sfac,
                      max_nbreak, bw, quiet=False):
    """
    Fit a B-spline curve, parameterized by freqs, to proj_port.

    Returns (tck, u, fp, ier, msg); see si.splprep(...) for details.  Returns
        ([np.array([]), np.array([]), 0], np.array([]), None, None, None) if
        proj_port has no columns (i.e. ncomp == 0).

    proj_port is an nchan x ncomp array of projections of profiles onto
        ncomp basis eigenvectors.
    freqs is the nchan array of frequencies corresponding to proj_port,
        increasing or decreasing monotonically with bw's sign.
    spl_weights are the nchan weights passed to si.splprep(...) as w.
    noise_stds are the nchan noise levels used to construct the default
        smoothing condition s (see sfac below).
    k is the polynomial degree of the spline; see make_spline_model(...).
    sfac is a multiplicative smoothing factor; see make_spline_model(...).
    max_nbreak is the maximum number of breakpoints (unique knots) to allow;
        see make_spline_model(...).
    bw is the bandwidth of the data (its sign determines whether freqs needs
        to be reversed for si.splprep(...), which requires u increasing).
    quiet=True suppresses output.
    """
    ncomp = proj_port.shape[1]
    if ncomp == 0:
        return [np.array([]), np.array([]), 0], np.array([]), None, None, None
    nu_lo, nu_hi = freqs.min(), freqs.max()
    s = sfac * len(proj_port) * np.sum((spl_weights * noise_stds)**2) / \
            sum(spl_weights)**2
    if bw < 0: flip = -1   #u in si.splprep has to be increasing...
    else: flip = 1
    #Find the B-spline curve traced by the projected vectors,
    #parameterized by frequency
    (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
            w=spl_weights[::flip], u=freqs[::flip], ub=nu_lo, ue=nu_hi,
            k=k, task=0, s=s, t=None, full_output=1, nest=None, per=0,
            quiet=int(quiet))
    if max_nbreak is not None and len(np.unique(tck[0])) > max_nbreak:
        if max_nbreak < 2:
            print("max_nbreak not >= 2; setting max_nbreak = 2...")
            max_nbreak = 2
        if max_nbreak == 2: s = np.inf
        (tck,u), fp, ier, msg = si.splprep(proj_port[::flip].T,
                w=spl_weights[::flip], u=freqs[::flip], ub=nu_lo,
                ue=nu_hi, k=k, task=0, s=s, t=None, full_output=1,
                nest=max_nbreak+(k*2), per=0, quiet=int(quiet))
    if ier > 1: #Will also catch when ier == "unknown"
        print("Something went wrong in si.splprep:\n%s" % msg)
    return tck, u, fp, ier, msg


class DataPortrait(DataPortrait):
    """
    DataPortrait is a class that contains the data to which a model is fit.

    This class adds methods and attributes to the parent class specific to
        modeling profile evolution with a B-spline curve.
    """

    def make_spline_model(self, max_ncomp=10, smooth=True, snr_cutoff=150.0,
                          rchi2_tol=0.1, k=3, sfac=1.0, max_nbreak=None,
                          wavelet='db8', model_name=None, quiet=False,
                          **kwargs):
        """
        Make a model based on PCA and B-spline interpolation.

        max_ncomp is the maximum number of PCA components to use in the
            B-spline parameterization; max_ncomp <= 10.
        smooth=True will smooth the eigenvectors and mean profile using
            a reduced chi-squared figure-of-merit.
        snr_cutoff is the S/N ratio value above or equal to which an
            eigenvector is deemed "significant".  Setting it equal to np.inf
            would ensure only a mean profile model is returned.
        rchi2_tol is the tolerance parameter that will allow greater deviations
            in the smoothed profile from the input profiles' shapes.
        k is the polynomial degree of the spline; cubic splines (k=3)
            recommended; 1 <= k <= 5.  NB: polynomial order = degree + 1.
        sfac is a multiplicative smoothing factor passed to si.splprep; greater
            values result in more smoothing.  sfac=0 will make an interpolating
            model anchored on the input data profiles.
        max_nbreak is the maximum number of breakpoints (unique knots) to
            allow.  If provided, this may override sfac and enforce smoothing
            based on max_nbreak breakpoints.  That is, if the fit returns n >
            max_nbreak breakpoints, it will refit using maximum max_nbreak
            breakpoints, irrespective of the other smoothing condition.  The
            corresponding maximum number of B-splines will be max_nspline =
            max_nbreak + k - 1.  max_nbreak should be >= 2.
        wavelet is the name of the mother wavelet used (via smart_smooth(...))
            to smooth the mean profile and eigenprofiles when smooth=True; see
            wavelet_smooth(...) for more [default='db8'].
        model_name is the name of the model; defaults to self.datafile +
            '.spl'
        quiet=True suppresses output.
        **kwargs get passed to find_significant_eigvec(...).
        """

        # Definitions
        port = self.portx
        pca_weights = old_div(self.SNRsxs, np.sum(self.SNRsxs))
        freqs = self.freqsxs[0]
        # Check nbin
        nbin = port.shape[1]
        if nbin % 2 != 0:
            print("nbin = %d is odd; cannot wavelet_smooth.\n" % nbin)
            smooth = False
        elif np.modf(np.log2(nbin))[0] != 0.0:
            print(
                "nbin = %d is not a power of two; can only try wavelet_smooth to one level; recommend resampling to a power-of-two number of phase bins.\n" % nbin)
        # Do principal component analysis and (optionally) smooth the mean
        # profile and significant eigenprofiles
        mean_prof, eigval, eigvec, ieig, ncomp, smooth_mean_prof, \
                smooth_eigvec = _fit_pca_and_smooth(port, freqs, pca_weights,
                        max_ncomp, smooth, snr_cutoff, rchi2_tol, wavelet,
                        quiet=quiet, **kwargs)

        if ncomp == 0:  # Will make model with constant average port
            proj_port = port[:, :ncomp]
            if smooth:
                modelx = reconst_port = np.tile(smooth_mean_prof,
                                                len(freqs)).reshape(len(freqs), port.shape[1])
                model = np.tile(smooth_mean_prof,
                                len(self.freqs[0])).reshape(len(self.freqs[0]),
                                                            port.shape[1])
            else:
                modelx = reconst_port = np.tile(mean_prof,
                                                len(freqs)).reshape(len(freqs), port.shape[1])
                model = np.tile(mean_prof,
                                len(self.freqs[0])).reshape(len(self.freqs[0]),
                                                            port.shape[1])
        else:
            delta_port = port - mean_prof
            if smooth:
                reconst_port = reconstruct_portrait(port, mean_prof,
                                                    smooth_eigvec[:, ieig])
                # Find the projections of the profiles onto the basis components
                proj_port = np.dot(delta_port, smooth_eigvec[:, ieig])
            else:
                reconst_port = reconstruct_portrait(port, mean_prof,
                                                    eigvec[:, ieig])
                # Find the projections of the profiles onto the basis components
                proj_port = np.dot(delta_port, eigvec[:, ieig])

        spl_weights = pca_weights
        tck, u, fp, ier, msg = _fit_spline_curve(proj_port, freqs,
                spl_weights, self.noise_stdsxs, k, sfac, max_nbreak, self.bw,
                quiet=quiet)

        # Build model
        if ncomp != 0:
            if smooth:
                modelx = gen_spline_portrait(smooth_mean_prof, freqs,
                                             smooth_eigvec[:, ieig], tck)
                model = gen_spline_portrait(smooth_mean_prof, self.freqs[0],
                                            smooth_eigvec[:, ieig], tck)
            else:
                modelx = gen_spline_portrait(mean_prof, freqs, eigvec[:, ieig],
                                             tck)
                model = gen_spline_portrait(mean_prof, self.freqs[0],
                                            eigvec[:, ieig], tck)

        # Assign new attributes
        self.ieig = ieig
        self.ncomp = ncomp
        self.eigvec = eigvec
        self.eigval = eigval
        self.mean_prof = mean_prof
        if smooth:
            self.smooth_mean_prof = smooth_mean_prof
            self.smooth_eigvec = smooth_eigvec
        self.proj_port = proj_port
        self.reconst_port = reconst_port
        # tck contains the knot locations t, B-spline coefficients c, and
        # polynomial degree k -- end knots will have multiplicity k+1, interior
        # breakpoints will have multiplicity 1 for maximum continuity.  The
        # number of B-splines will be n = l + k, where l is the number of
        # intervals.  l = number of breakpoints - 1 = number of unique knots - 1
        # = len(tck[0]) - 2*tck[2] - 1.
        self.tck, self.u, self.fp, self.ier, self.msg = tck, u, fp, ier, msg
        if model_name is None:
            self.model_name = self.datafile + '.spl'
        else:
            self.model_name = model_name
        self.model = model
        self.modelx = modelx
        self.model_masked = self.model * self.masks[0, 0]

        if not quiet:
            if proj_port.sum():
                print(
                    "B-spline interpolation model %s uses %d basis profile components and %d breakpoints (%d B-splines with k=%d)." % (
                    self.model_name, ncomp,
                    len(np.unique(self.tck[0])),
                    len(self.tck[0]) - self.tck[2] - 1, self.tck[2]))
            else:
                print(
                    "B-spline interpolation model %s uses 0 basis profile components; it returns the average profile." % (
                        self.model_name))

    def write_model(self, outfile, quiet=False):
        """
        Write the output (pickle file) model to outfile.
        """
        of = open(outfile, "wb")
        if hasattr(self, "smooth_eigvec"):
            if len(self.ieig):
                pickle.dump([self.model_name, self.source, self.datafile,
                             self.smooth_mean_prof, self.smooth_eigvec[:, self.ieig],
                             self.tck], of, protocol=2)
            else:
                pickle.dump([self.model_name, self.source, self.datafile,
                             self.smooth_mean_prof, self.smooth_eigvec[:, []],
                             self.tck], of, protocol=2)
        else:
            if len(self.ieig):
                pickle.dump([self.model_name, self.source, self.datafile,
                             self.mean_prof, self.eigvec[:, self.ieig], self.tck], of, protocol=2)
            else:
                pickle.dump([self.model_name, self.source, self.datafile,
                             self.mean_prof, self.eigvec[:, []], self.tck], of,
                             protocol=2)

        of.close()
        if not quiet:
            print("Wrote modelfile %s." % outfile)

    def show_eigenprofiles(self, ncomp=None, title=None, **kwargs):
        """
        Calls show_eigenprofiles(...) to make plots of mean/eigen profiles.

        see show_eigenprofiles(...) for details.

        ncomp=None plots self.ncomp PCA components, otherwise plots the number
            of components specified.
        **kwargs get passed to show_eigenprofiles(...).
        """
        if ncomp is None: ncomp = self.ncomp
        if hasattr(self, "smooth_eigvec"):
            if ncomp:
                eigvec = self.eigvec[:, self.ieig[:ncomp]].T
                seigvec = self.smooth_eigvec[:, self.ieig[:ncomp]].T
            else:
                eigvec = None
                seigvec = None
            show_eigenprofiles(eigvec, seigvec, self.mean_prof,
                               self.smooth_mean_prof, title=title, **kwargs)
        else:
            if ncomp:
                eigvec = self.eigvec[:, self.ieig[:ncomp]].T
            else:
                eigvec = None
            show_eigenprofiles(eigvec, None, self.mean_prof, None, title=title,
                               **kwargs)

    def show_spline_curve_projections(self, ncomp=None, title=None, **kwargs):
        """
        Calls show_spline_curve_projections(...) to make plots of the model.

        see show_spline_curve_projections(...) for details.

        ncomp=None plots self.ncomp PCA components, otherwise plots the number
            of components specified.
        **kwargs get passed to show_spline_curve_projections(...).
        """
        if ncomp is None: ncomp = self.ncomp
        if ncomp:
            show_spline_curve_projections(self.proj_port, self.tck,
                                          self.freqsxs[0], old_div(self.SNRsxs, np.sum(self.SNRsxs)),
                                          ncoord=ncomp, title=title, **kwargs)

    def select_wavelet(self, wavelets=('db4', 'db8', 'db12', 'db20', 'sym8',
                                       'sym12', 'coif4'), rchi2_tol=0.1,
                       quiet=False):
        """
        Pick the wavelet that best denoises the (full-dataset) mean profile.

        For each candidate wavelet, the mean profile (as used by
            make_spline_model(...)) is put through the same per-profile
            search smart_smooth(...) uses (decomposition level and
            threshold factor chosen to maximize a pseudo-S/N subject to
            keeping the reduced chi-squared within rchi2_tol of 1.0).  Among
            the wavelets that satisfy that constraint, the one selected is
            the one whose *residual* (mean profile minus smoothed profile)
            looks most like unstructured white noise: whatever a genuinely
            clean denoising fails to capture in the smooth curve should be
            indistinguishable from noise, not leftover correlated structure
            (the wavelet under-resolved a real feature) or ringing (a
            reconstruction artefact).  This is quantified by how close the
            residual's zero-crossing rate (see pplib.count_crossings(...))
            is to the 0.5 expected for symmetric white noise -- too FEW
            crossings means real structure was left behind; too MANY is
            atypical of pure noise and can catch some kinds of artefacts.

            An earlier version of this method ranked wavelets by the same
            pseudo-S/N used to pick level/factor within a single wavelet.
            That turned out to scale with a wavelet's filter length almost
            independent of how well it actually denoises -- longer-support
            wavelets structurally leave less high-frequency content in the
            reconstruction, which inflates that ratio without reflecting
            better denoising -- so it could not fairly rank different
            wavelet families against each other, and is not used here.

        Returns (best_wavelet, info), where info is a
            {wavelet: DataBunch(snr, residual_crossing_frac)} dict for every
            candidate (snr is the pseudo-S/N noted above, kept only for
            inspection; residual_crossing_frac is None for any wavelet that
            could not keep the reduced chi-squared within tolerance at any
            decomposition level, and such wavelets are never selected).
            Falls back to 'db8' (or wavelets[0], if 'db8' is not a
            candidate) if every wavelet fails that way.

        wavelets is the list of candidate mother wavelets to try.
        rchi2_tol is as in make_spline_model(...).
        quiet=True suppresses output.
        """
        if 'pywt' not in sys.modules:
            raise ImportError("You failed to import pywt and need PyWavelets to use select_wavelet!")

        port = self.portx
        pca_weights = old_div(self.SNRsxs, np.sum(self.SNRsxs))
        mean_prof = old_div((port.T * pca_weights).T.sum(axis=0),
                            pca_weights.sum())
        nbin = port.shape[1]
        try_nlevels = _default_try_nlevels(nbin)

        info = {}
        for w in wavelets:
            smooth_prof, snr, unused_ilevel, unused_fact = \
                    _smart_smooth_profile(mean_prof, w, rchi2_tol,
                            try_nlevels)
            if snr > 0.0:
                residual = mean_prof - smooth_prof
                crossing_frac = old_div(count_crossings(residual, 0.0),
                                        float(nbin - 1))
            else:
                crossing_frac = None
            info[w] = DataBunch(snr=snr, residual_crossing_frac=crossing_frac)
            if not quiet:
                if crossing_frac is None:
                    print("select_wavelet: wavelet '%s' could not keep the mean profile's reduced chi2 within tolerance at any level." % w)
                else:
                    print("select_wavelet: wavelet '%s' residual zero-crossing fraction = %.4f (0.5 = ideal white-noise residual)." % (
                            w, crossing_frac))

        valid = [w for w in wavelets if info[w].residual_crossing_frac is not None]
        if not valid:
            best_wavelet = 'db8' if 'db8' in wavelets else wavelets[0]
            if not quiet:
                print("select_wavelet: no candidate wavelet kept the mean profile's reduced chi2 within tolerance; falling back to '%s'." % best_wavelet)
        else:
            best_wavelet = min(valid,
                    key=lambda w: abs(info[w].residual_crossing_frac - 0.5))
            if not quiet:
                print("select_wavelet: selected wavelet '%s' (residual zero-crossing fraction = %.4f)." % (
                        best_wavelet, info[best_wavelet].residual_crossing_frac))
        return best_wavelet, info

    def photoshop_spline_model(self, max_ncomp=10, snr_cutoff=150.0,
                               rchi2_tol=0.1,
                               wavelets=('db4', 'db8', 'db12', 'db20', 'sym8',
                                         'sym12', 'coif4'),
                               ks=(1, 3, 5), max_nbreak_candidates=(None,),
                               test_frac=0.2, nrepeat=20, sfac_bounds=(-3, 2),
                               decouple_wavelet=True, seed=None, apply=True,
                               model_name=None, quiet=False, **kwargs):
        """
        Auto-select (wavelet, k, sfac, max_nbreak) for make_spline_model(...)
            via Monte Carlo cross-validation over frequency channels.

        If decouple_wavelet=True (default), a single wavelet is first chosen
            by select_wavelet(...) -- how well it denoises the full-dataset
            mean profile, independent of k/sfac/max_nbreak -- and only that
            wavelet is used below.  This turns the search from
            len(wavelets)*nrepeat expensive PCA/eigenprofile fits into
            roughly len(wavelets) cheap single-profile fits (select_wavelet)
            plus nrepeat expensive fits (for the one chosen wavelet), at the
            cost of assuming the wavelet that best smooths the mean profile
            also serves the (lower-S/N) eigenprofiles well.  With
            decouple_wavelet=False, wavelet is instead cross-validated
            jointly with k/sfac/max_nbreak below (len(wavelets)*nrepeat
            expensive fits), as in earlier versions of this method.

        For each candidate wavelet (only one, if decouple_wavelet=True),
            nrepeat random train/test splits of the (non-edge) frequency
            channels are drawn.  For each split, PCA and eigenprofile/mean-
            profile smoothing are fit on the training channels only -- this
            is the expensive step, and is independent of k, sfac, and
            max_nbreak, so it is done once per (wavelet, split) and reused.
            Then, for each (k, max_nbreak) combination, sfac is optimized
            (on a log scale) to minimize the out-of-sample reduced
            chi-squared of the B-spline curve evaluated on the held-out
            channels.  The standard error of that score across the nrepeat
            splits is used to apply a "1-SE rule": among (wavelet, k,
            max_nbreak, sfac) combinations statistically indistinguishable
            from the minimum-scoring one, the most regularized (largest
            sfac, then smallest max_nbreak, then k == 3, then
            wavelet == 'db8') is selected.

        If apply=True (default), make_spline_model(...) is then called on the
            full dataset using the selected hyperparameters, exactly as if
            they had been supplied by hand.  self.photoshop_results holds the
            full grid of (wavelet, k, max_nbreak, sfac, score, se) that was
            evaluated, for inspection; self.photoshop_best holds the
            selected entry; self.photoshop_wavelet_scores holds the
            select_wavelet(...) scores, if decouple_wavelet=True.

        max_ncomp, snr_cutoff, rchi2_tol are as in make_spline_model(...) and
            are held fixed here (not tuned).
        wavelets is the list of candidate mother wavelets to try.  This is a
            curated shortlist, not pywt.wavelist()'s full set, to keep the
            (expensive) PCA/smoothing stage's cost bounded; override for an
            exhaustive search.
        ks is the list of candidate spline polynomial degrees to try.
        max_nbreak_candidates is the list of candidate max_nbreak caps to try
            (None means unconstrained, i.e. governed only by sfac).
        test_frac is the fraction of (non-edge) channels held out per split.
        nrepeat is the number of random train/test splits per wavelet.
        sfac_bounds are the (log10(sfac_min), log10(sfac_max)) search bounds
            passed to the underlying opt.brute search over sfac.
        decouple_wavelet is described above.
        seed seeds the random number generator, for reproducibility.
        apply=True fits and stores the final model using the chosen
            hyperparameters; apply=False only computes/stores the search
            results without changing self's model attributes.
        model_name is passed to make_spline_model(...) if apply=True.
        quiet=True suppresses output.
        **kwargs get passed to find_significant_eigvec(...) (and hence to
            make_spline_model(...), if apply=True).
        """
        if 'pywt' not in sys.modules:
            raise ImportError("You failed to import pywt and need PyWavelets to use photoshop_spline_model!")

        port = self.portx
        freqs = self.freqsxs[0]
        SNRs = self.SNRsxs
        noise_stds = self.noise_stdsxs
        nchanx, nbin = port.shape
        rng = np.random.RandomState(seed)

        if decouple_wavelet:
            chosen_wavelet, wavelet_scores = self.select_wavelet(
                    wavelets=wavelets, rchi2_tol=rchi2_tol, quiet=quiet)
            self.photoshop_wavelet_scores = wavelet_scores
            search_wavelets = (chosen_wavelet,)
        else:
            search_wavelets = wavelets

        # Never hold out the band-edge channels -- doing so would shrink the
        # training fit's ub/ue boundary and force gen_spline_portrait to
        # extrapolate for that channel, confounding "bad hyperparameter" with
        # "channel required extrapolation".
        hold_pool = np.array([ichan for ichan in range(nchanx)
                if ichan not in (freqs.argmin(), freqs.argmax())])
        n_test = max(1, int(round(test_frac * len(hold_pool))))

        # Expensive stage: PCA + smoothing, once per (wavelet, repeat);
        # reused below across all (k, max_nbreak, sfac) combinations.
        stash = {}
        for w in search_wavelets:
            if not quiet:
                print("photoshop_spline_model: fitting %d train/test splits for wavelet '%s'..." % (nrepeat, w))
            for r in range(nrepeat):
                test_idx = rng.choice(hold_pool, n_test, replace=False)
                train_idx = np.setdiff1d(np.arange(nchanx), test_idx)
                port_train = port[train_idx]
                freqs_train = freqs[train_idx]
                pca_weights_train = old_div(SNRs[train_idx], np.sum(SNRs[train_idx]))
                mean_prof, eigval, eigvec, ieig, ncomp, smooth_mean_prof, \
                        smooth_eigvec = _fit_pca_and_smooth(port_train,
                                freqs_train, pca_weights_train, max_ncomp,
                                True, snr_cutoff, rchi2_tol, w, quiet=True,
                                **kwargs)
                stash[(w, r)] = DataBunch(
                        proj_port_train=np.dot(port_train - mean_prof,
                            smooth_eigvec[:, ieig]),
                        freqs_train=freqs_train,
                        spl_weights=pca_weights_train,
                        noise_train=noise_stds[train_idx],
                        smooth_mean_prof=smooth_mean_prof,
                        smooth_eigvec=smooth_eigvec, ieig=ieig,
                        freqs_test=freqs[test_idx], port_test=port[test_idx],
                        noise_test=noise_stds[test_idx])

        # Cheap stage: optimize sfac (log10) per (wavelet, k, max_nbreak).
        results = []
        for w in search_wavelets:
            for k in ks:
                for max_nbreak in max_nbreak_candidates:

                    def per_repeat_scores(log10_sfac):
                        sfac = 10.0**log10_sfac
                        scores = np.zeros(nrepeat)
                        for r in range(nrepeat):
                            d = stash[(w, r)]
                            tck, u, fp, ier, msg = _fit_spline_curve(
                                    d.proj_port_train, d.freqs_train,
                                    d.spl_weights, d.noise_train, k, sfac,
                                    max_nbreak, self.bw, quiet=True)
                            model_test = gen_spline_portrait(
                                    d.smooth_mean_prof, d.freqs_test,
                                    d.smooth_eigvec[:, d.ieig], tck)
                            # dof=1 leaves the raw (unreduced) summed chi2,
                            # so it can be normalized once below by the true
                            # dof (nbin per held-out channel).
                            scores[r] = old_div(get_red_chi2(d.port_test,
                                    model_test, errs=d.noise_test, dof=1),
                                    (nbin * len(d.port_test)))
                        return scores

                    def cv_score(log10_sfac):
                        return per_repeat_scores(log10_sfac).mean()

                    brute_result = opt.brute(cv_score, ranges=[sfac_bounds],
                            Ns=25, full_output=True)
                    best_log10_sfac = brute_result[0][0]
                    scores = per_repeat_scores(best_log10_sfac)
                    score = scores.mean()
                    se = old_div(scores.std(ddof=1), np.sqrt(nrepeat)) \
                            if nrepeat > 1 else 0.0
                    results.append(DataBunch(wavelet=w, k=k,
                            max_nbreak=max_nbreak, sfac=10.0**best_log10_sfac,
                            score=score, se=se))

        # Selection: minimum CV score, then a 1-SE-rule tie-break that
        # prefers more strongly regularized choices among combinations
        # statistically indistinguishable from the minimum.
        global_best = min(results, key=lambda res: res.score)
        threshold = global_best.score + global_best.se
        candidates = [res for res in results if res.score <= threshold]

        def regularization_key(res):
            nbreak_val = res.max_nbreak if res.max_nbreak is not None \
                    else np.inf
            wavelet_rank = 0 if res.wavelet == 'db8' else \
                    pw.Wavelet(res.wavelet).dec_len
            return (-res.sfac, nbreak_val, res.k != 3, wavelet_rank)
        best = min(candidates, key=regularization_key)

        self.photoshop_results = results
        self.photoshop_best = best

        if not quiet:
            print("photoshop_spline_model: selected wavelet=%s, k=%d, sfac=%.4g, max_nbreak=%s (CV reduced chi2 = %.4f +/- %.4f; global minimum was %.4f +/- %.4f)." % (
                    best.wavelet, best.k, best.sfac, str(best.max_nbreak),
                    best.score, best.se, global_best.score, global_best.se))

        if apply:
            self.make_spline_model(max_ncomp=max_ncomp, smooth=True,
                    snr_cutoff=snr_cutoff, rchi2_tol=rchi2_tol, k=best.k,
                    sfac=best.sfac, max_nbreak=best.max_nbreak,
                    wavelet=best.wavelet, model_name=model_name,
                    quiet=quiet, **kwargs)

        return best.wavelet, best.k, best.sfac, best.max_nbreak


if __name__ == "__main__":

    from optparse import OptionParser

    usage = "Usage: %prog -d <datafile> [options]"
    parser = OptionParser(usage)
    # parser.add_option("-h", "--help",
    #                  action="store_true", dest="help", default=False,
    #                  help="Show this help message and exit.")
    parser.add_option("-d", "--datafile",
                      action="store", metavar="archive", dest="datafile",
                      help="PSRCHIVE archive from which to make model, or a metafile listing multiple archives (i.e., from different bands).  If providing a metafile, the achives must already be aligned.")
    parser.add_option("-o", "--modelfile",
                      action="store", metavar="modelfile", dest="modelfile",
                      help="Name for output model (pickle) file. [default=datafile.spl].")
    parser.add_option("-l", "--model_name",
                      action="store", metavar="model_name", dest="model_name",
                      default=None,
                      help="Optional name for model [default=datafile.spl].")
    parser.add_option("-a", "--archive",
                      action="store", metavar="archive", dest="archive",
                      default=None,
                      help="Name for optional output PSRCHIVE archive.  Will work only if the input is a single archive.")
    parser.add_option("-N", "--norm",
                      action="store", metavar="normalization", dest="norm",
                      default="prof",
                      help="Normalize the input data by channel ('None', 'mean', 'max' (not recommended), 'rms' (off-pulse noise), 'prof' (mean profile flux) [default], or 'abs' (sqrt{vector modulus})).")
    parser.add_option("-s", "--smooth",
                      action="store_true", metavar="smooth", dest="smooth",
                      default=False,
                      help="Smooth the eigenvectors and mean profile [recommended] using default wavelet_smooth options and smart_smooth.")
    parser.add_option("-n", "--max_ncomp",
                      action="store", metavar="max_ncomp", dest="max_ncomp",
                      default=10,
                      help="Maximum number of principal components to use in PCA reconstruction of the data.  max_ncomp is limited to a maximum of 10 by the B-spline representation in scipy.interpolate.")
    parser.add_option("-S", "--snr",
                      action="store", metavar="snr_cutoff", dest="snr_cutoff",
                      default=150.0,
                      help="S/N ratio cutoff for determining 'significant' eigenprofiles.  A value somewhere over 100.0 should be good. [default=150.0].")
    parser.add_option("-T", "--rchi2_tol",
                      action="store", metavar="tolerance", dest="rchi2_tol",
                      default=0.1,
                      help="Tweak this between 0.0 and 0.1 [default] if the returned eigenprofiles are not smooth enough.")
    parser.add_option("-k", "--degree",
                      action="store", metavar="degree", dest="k", default=3,
                      help="Degree of the spline.  Cubic splines (k=3) are recommended [default]. 1 <= k <=5.")
    parser.add_option("-f", "--sfac",
                      action="store", metavar="smooth_factor", dest="sfac",
                      default=1.0,
                      help="To change the smoothness of the B-spline model, tweak this between 0.0 (interpolating spline that passes through all data points) and a large number (guarantees maximum two breakpoints = maximum smoothness).  Alternatively, use -t.")
    parser.add_option("-t", "--knots",
                      action="store", metavar="max_knots", dest="max_nbreak",
                      default=None,
                      help="The maximum number of unique knots.  This functions esentially as an ignorant smoothing condition in case the default settings return a fit with more than max_knots number of unique knots in the spline model.  e.g., 10 unique knots are more than usually necessary.")
    parser.add_option("--photoshop",
                      action="store_true", dest="photoshop", default=False,
                      help="Auto-select wavelet, k (-k), sfac (-f), and max_nbreak (-t) via cross-validation, instead of using the given/default values for those options.  Implies -s.")
    parser.add_option("--photoshop-wavelets",
                      action="store", metavar="wavelets", dest="photoshop_wavelets",
                      default="db4,db8,db12,db20,sym8,sym12,coif4",
                      help="Comma-separated list of candidate wavelets to try with --photoshop. [default=db4,db8,db12,db20,sym8,sym12,coif4].")
    parser.add_option("--photoshop-repeats",
                      action="store", metavar="nrepeat", dest="photoshop_repeats",
                      default=20,
                      help="Number of random train/test splits per wavelet used by --photoshop. [default=20].")
    parser.add_option("--photoshop-testfrac",
                      action="store", metavar="test_frac", dest="photoshop_testfrac",
                      default=0.2,
                      help="Fraction of channels held out per split used by --photoshop. [default=0.2].")
    parser.add_option("--photoshop-seed",
                      action="store", metavar="seed", dest="photoshop_seed",
                      default=None,
                      help="Random seed for --photoshop, for reproducibility. [default=None].")
    parser.add_option("--plots",
                      action="store_true", dest="make_plots", default=False,
                      help="Save some plots related to the model with basename model_name (-l).")
    parser.add_option("--quiet",
                      action="store_true", dest="quiet", default=False,
                      help="Suppresses output.")

    (options, args) = parser.parse_args()

    if (options.datafile is None):
        print("\nppspline.py - make a pulse portrait model using PCA & B-spline interpolation\n")
        parser.print_help()
        print("")
        parser.exit()

    datafile = options.datafile
    modelfile = options.modelfile
    model_name = options.model_name
    archive = options.archive
    norm = options.norm
    smooth = options.smooth
    max_ncomp = int(options.max_ncomp)
    snr_cutoff = float(options.snr_cutoff)
    rchi2_tol = float(options.rchi2_tol)
    k = int(options.k)
    sfac = float(options.sfac)
    if options.max_nbreak is not None:
        max_nbreak = int(options.max_nbreak)
    else:
        max_nbreak = None
    photoshop = options.photoshop
    photoshop_wavelets = tuple(options.photoshop_wavelets.split(","))
    photoshop_repeats = int(options.photoshop_repeats)
    photoshop_testfrac = float(options.photoshop_testfrac)
    if options.photoshop_seed is not None:
        photoshop_seed = int(options.photoshop_seed)
    else:
        photoshop_seed = None
    make_plots = options.make_plots
    quiet = options.quiet

    dp = DataPortrait(datafile, quiet=quiet)

    if norm in ("mean", "max", "prof", "rms", "abs"):
        dp.normalize_portrait(norm)

    if photoshop:
        if not quiet:
            print("--photoshop given; -k/-f/-t (if given) are ignored in favor of cross-validated values.")
        dp.photoshop_spline_model(max_ncomp=max_ncomp, snr_cutoff=snr_cutoff,
                rchi2_tol=rchi2_tol, wavelets=photoshop_wavelets,
                test_frac=photoshop_testfrac, nrepeat=photoshop_repeats,
                seed=photoshop_seed, model_name=model_name, quiet=quiet)
    else:
        dp.make_spline_model(max_ncomp=max_ncomp, smooth=smooth,
                             snr_cutoff=snr_cutoff, rchi2_tol=rchi2_tol, k=k, sfac=sfac,
                             max_nbreak=max_nbreak, model_name=model_name, quiet=quiet)

    if modelfile is None: modelfile = datafile + ".spl"
    dp.write_model(modelfile, quiet=quiet)

    if archive is not None and len(dp.datafiles) == 1:
        dp.write_model_archive(archive, quiet=quiet)

    if make_plots:
        dp.show_eigenprofiles(title=dp.model_name, savefig=dp.model_name)
        dp.show_spline_curve_projections(title=dp.model_name,
                                         savefig=dp.model_name)
        dp.show_model_fit(savefig=dp.model_name + '.resids.png')
