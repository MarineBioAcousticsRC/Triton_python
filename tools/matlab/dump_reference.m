function dump_reference(varargin)
%DUMP_REFERENCE  Generate MATLAB ground truth for the Triton-Python parity suite.
%
% Runs the *existing, unmodified* Triton base-folder code over the Phase 0 fixture corpus and
% writes its outputs to fixtures/reference/.  The Python implementation must then reproduce
% those outputs exactly.  This is the artefact that makes the port verifiable, and it has to
% be produced while MATLAB is still the source of truth.
%
% Usage
%   dump_reference                                  % auto-locate everything
%   dump_reference('triton', 'D:\Code\Triton-master')
%   dump_reference('fixtures','...','out','...')
%   dump_reference('sections', {'headers','dsp'})   % subset
%
% Name/value options
%   'triton'    path to Triton-master        (default: two dirs up + \Triton-master, or `which triton`)
%   'fixtures'  fixture directory            (default: <repo>\fixtures\generated)
%   'out'       output directory             (default: <repo>\fixtures\reference)
%   'sections'  cellstr subset of:
%               'headers'  x.wav header parse, via rdxwavhd AND io/ioReadXWAVHeader
%               'timing'   timestr/timenum round trips, wavname2dnum, datenum precision
%               'readseg'  sample blocks at chosen plot times, incl. across a gap
%               'dsp'      hanning, mkspecgram, pwelch, interp1 TF, logfmap
%               'ltsa'     header + data block for any .ltsa in the fixture directory
%
% Notes
%   * No Triton GUI is started.  tr_headless_handles builds the few HANDLES that core
%     functions touch, so readseg/check_time/mkspecgram run for real rather than being
%     re-implemented here.
%   * Everything floating point is written as raw little-endian binary plus a JSON sidecar.
%     JSON alone is not bit-exact and this suite exists to catch fractional-dB errors.
%
% Part of the Triton-Python Phase 0 parity tooling.

%% ---------------------------------------------------------------- arguments
p = inputParser;
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));                     % tools/matlab -> tools -> repo
addParameter(p,'triton','');
addParameter(p,'fixtures',fullfile(repo,'fixtures','generated'));
addParameter(p,'out',fullfile(repo,'fixtures','reference'));
addParameter(p,'sections',{'headers','timing','readseg','dsp','ltsa'});
parse(p,varargin{:});
opt = p.Results;

if isempty(opt.triton)
    w = which('triton');
    if ~isempty(w)
        opt.triton = fileparts(w);
    else
        opt.triton = fullfile(fileparts(repo),'Triton-master');
    end
end
if exist(fullfile(opt.triton,'rdxwavhd.m'),'file') ~= 2
    error('dump_reference:noTriton', ...
        'Triton base folder not found at %s (pass ''triton'',<path>)', opt.triton);
end
addpath(opt.triton);
addpath(fullfile(opt.triton,'io'));    % for the ioReadXWAVHeader cross-check
addpath(here);

if ~exist(opt.out,'dir'); mkdir(opt.out); end
fprintf('Triton   : %s\n', opt.triton);
fprintf('Fixtures : %s\n', opt.fixtures);
fprintf('Output   : %s\n\n', opt.out);

global PARAMS HANDLES DATA %#ok<GVMIS>

%% ---------------------------------------------------------------- provenance
% built field by field: struct('a',{cellarray}) would create a struct *array*
env = struct();
env.matlab_version = version;
env.matlab_release = version('-release');
env.computer       = computer;
env.generated      = datestr(now,'yyyy-mm-ddTHH:MM:SS'); %#ok<TNOW1,DATST>
env.triton_path    = opt.triton;
env.fixtures_path  = opt.fixtures;
v = ver;
env.toolboxes      = arrayfun(@(t) sprintf('%s %s', t.Name, t.Version), v, ...
                              'UniformOutput', false);
tr_dump(opt.out,'_environment',env);

xw = dir(fullfile(opt.fixtures,'*.x.wav'));
wv = dir(fullfile(opt.fixtures,'*.wav'));
wv = wv(~contains({wv.name},'.x.wav'));       % plain wavs only
fprintf('Found %d x.wav and %d wav fixtures\n\n', numel(xw), numel(wv));

want = @(s) any(strcmpi(opt.sections,s));

%% ================================================================= headers
if want('headers')
    fprintf('--- headers ---\n');
    tr_headless_handles();
    % cell, not a struct array: the v2 header adds fields (drate/dt), so records are not
    % all the same shape
    out = {};
    for k = 1:numel(xw)
        PARAMS = [];
        PARAMS.inpath = opt.fixtures;
        PARAMS.infile = xw(k).name;
        rdxwavhd;                                   % populates PARAMS.xhd / PARAMS.raw

        rec = struct();
        rec.file  = xw(k).name;
        rec.xhd   = PARAMS.xhd;
        rec.fs    = PARAMS.fs;
        rec.nch   = PARAMS.nch;
        rec.nBits = PARAMS.nBits;
        rec.samp_byte = PARAMS.samp.byte;
        rec.xgain = PARAMS.xgain;
        % datenums are floats -- keep full precision as hex as well as decimal
        rec.raw_dnumStart_hex = arrayfun(@(v) sprintf('%bx',v), ...
                                         PARAMS.raw.dnumStart,'UniformOutput',false);
        rec.raw_dnumEnd_hex   = arrayfun(@(v) sprintf('%bx',v), ...
                                         PARAMS.raw.dnumEnd,'UniformOutput',false);
        rec.start_dnum_hex    = sprintf('%bx', PARAMS.start.dnum);
        rec.end_dnum_hex      = sprintf('%bx', PARAMS.end.dnum);
        rec.raw_dvecStart     = PARAMS.raw.dvecStart;
        rec.raw_dvecEnd       = PARAMS.raw.dvecEnd;

        % independent cross-check: io/ioReadXWAVHeader returns a struct, no globals.
        % If these two ever disagree we want to know before writing Python.
        try
            hdr2 = ioReadXWAVHeader(fullfile(opt.fixtures,xw(k).name));
            rec.io_agrees = isequal(hdr2.xhd.NumOfRawFiles, PARAMS.xhd.NumOfRawFiles) && ...
                            isequal(hdr2.xhd.byte_loc,      PARAMS.xhd.byte_loc) && ...
                            isequal(hdr2.xhd.byte_length,   PARAMS.xhd.byte_length) && ...
                            isequal(hdr2.xhd.sample_rate,   PARAMS.xhd.sample_rate);
        catch ME
            rec.io_agrees = false;
            rec.io_error  = ME.message;
        end

        % exact float dump of every derived datenum
        tr_dump(opt.out, sprintf('dnums__%s', safename(xw(k).name)), ...
                [PARAMS.raw.dnumStart(:), PARAMS.raw.dnumEnd(:)], 'binary');

        out{end+1} = rec; %#ok<AGROW>
        fprintf('  %-44s nrf=%d fs=%d %s\n', xw(k).name, PARAMS.xhd.NumOfRawFiles, ...
                PARAMS.fs, ternary(rec.io_agrees,'(io agrees)','(IO MISMATCH!)'));
    end
    tr_dump(opt.out,'xwav_headers',out);
end

%% ================================================================= timing
if want('timing')
    fprintf('--- timing ---\n');
    tr_headless_handles();

    % timestr for all 8 output types, over times that stress the ms/us rollover in
    % timestr.m:73-88 (the mmm==1000 correction).
    probe_dvecs = [ ...
        11  1 30  8 45  0.000; ...
        11  1 30  8 45  0.125; ...
        11  1 30  8 45  1.9999995; ...   % rounds up through the second boundary
        11  1 30 23 59 59.9999995; ...   % ... and through the day boundary
        11  1 30  8 45 12.345678; ...
        11 12 31 23 59 59.999; ...
         0  1  1  0  0  0.000];
    ts = cell(size(probe_dvecs,1), 8);
    for i = 1:size(probe_dvecs,1)
        for t = 1:8
            ts{i,t} = timestr(datenum(probe_dvecs(i,:)), t);
        end
    end
    tr_dump(opt.out,'timestr', struct('dvecs',probe_dvecs,'strings',{ts}));
    tr_dump(opt.out,'timestr_dnums', datenum(probe_dvecs), 'binary');

    % timenum is the inverse; check the round trip Triton itself relies on
    rt = cell(size(probe_dvecs,1),1);
    for i = 1:size(probe_dvecs,1)
        rt{i} = timenum(timestr(datenum(probe_dvecs(i,:)),1), 1);
    end
    tr_dump(opt.out,'timenum_roundtrip', cell2mat(rt), 'binary');

    % wavname2dnum over all six filename patterns it claims to support
    names = { ...
        'SOCAL41N_DL29_110130-084500.wav', ...
        'SOCAL41N_DL29_110130T084500Z.wav', ...
        'SOCAL41N_DL29_110130084500.wav', ...
        'SOCAL41N_DL29_20110130_084500.wav', ...
        'SOCAL41N_DL29_110130_084500.wav', ...
        'AMAR613.20110130T084500Z.wav'};
    dn = nan(numel(names),1);
    for i = 1:numel(names)
        v = wavname2dnum(names{i}, 0);
        if ~isempty(v); dn(i) = v(1); end
    end
    tr_dump(opt.out,'wavname2dnum', struct('names',{names}));
    tr_dump(opt.out,'wavname2dnum_dnums', dn, 'binary');

    % the precision argument from docs/formats/timebase.md, measured rather than asserted
    prec = struct( ...
        'eps_shifted_2011_days', eps(datenum([11 1 30 0 0 0])), ...
        'eps_true_2011_days',    eps(datenum([2011 1 30 0 0 0])), ...
        'eps_shifted_2011_us',   eps(datenum([11 1 30 0 0 0]))*86400*1e6, ...
        'eps_true_2011_us',      eps(datenum([2011 1 30 0 0 0]))*86400*1e6, ...
        'sample_period_us_at_200kHz', 1e6/200000);
    tr_dump(opt.out,'datenum_precision',prec);
    fprintf('  datenum eps: shifted %.4g us, true %.4g us, sample at 200kHz = 5 us\n', ...
            prec.eps_shifted_2011_us, prec.eps_true_2011_us);
end

%% ================================================================= readseg
if want('readseg')
    fprintf('--- readseg ---\n');
    index = {};
    for k = 1:numel(xw)
        cases = readseg_cases(opt.fixtures, xw(k).name);
        for c = 1:numel(cases)
            index{end+1} = run_readseg(opt, xw(k).name, cases(c)); %#ok<AGROW>
        end
    end
    tr_dump(opt.out,'readseg_index',index);
end

%% ================================================================= dsp
if want('dsp')
    fprintf('--- dsp ---\n');
    tr_headless_handles();

    % --- window definition.  This is the trap in docs/formats/ltsa.md 5.1: MATLAB
    % hanning(N) is the symmetric Hann window with its zero endpoints removed, which is
    % neither scipy hann(N) nor hann(N,sym=False).
    for N = [8 9 16 256 1000 1024]
        tr_dump(opt.out, sprintf('hanning_%d',N), hanning(N), 'binary');
    end
    tr_dump(opt.out,'window_note', struct('matlab_call','hanning(N)', ...
        'scipy_equivalent','scipy.signal.windows.hann(N+2, sym=True)[1:-1]'));

    % --- spectrogram, through Triton's own mkspecgram, on real fixture samples
    ref = pick_fixture(xw, 'xwav_v1_cont_1ch_16b_10k');
    if ~isempty(ref)
        for cfg = struct('nfft',{256,1000,1000,512}, 'ol',{0,0,50,75})
            load_segment(opt, ref, 0, 2.0);           % 2 s from the start of the file
            PARAMS.nfft    = cfg.nfft;
            PARAMS.overlap = cfg.ol;
            PARAMS.freq0   = 0;
            PARAMS.freq1   = PARAMS.fs/2;
            PARAMS.ch      = 1;
            mkspecgram;                                % sets PARAMS.pwr / .f / .t
            tag = sprintf('specgram_nfft%d_ol%d', cfg.nfft, cfg.ol);
            tr_dump(opt.out, [tag '__pwr'], PARAMS.pwr, 'binary');
            tr_dump(opt.out, [tag '__f'],   PARAMS.f(:), 'binary');
            tr_dump(opt.out, [tag '__t'],   PARAMS.t(:), 'binary');
            tr_dump(opt.out, [tag '__meta'], struct( ...
                'file',ref,'nfft',cfg.nfft,'overlap',cfg.ol,'fs',PARAMS.fs, ...
                'freq0',PARAMS.freq0,'freq1',PARAMS.freq1, ...
                'fimin',PARAMS.fimin,'fimax',PARAMS.fimax, ...
                'plot_start_offset_sec',0,'duration_sec',2.0, ...
                'source','mkspecgram.m'));
            fprintf('  specgram nfft=%4d ol=%2d%%  -> %dx%d\n', cfg.nfft, cfg.ol, ...
                    size(PARAMS.pwr,1), size(PARAMS.pwr,2));
        end

        % --- pwelch exactly as calc_ltsa.m calls it, including the int8 quantisation
        load_segment(opt, ref, 0, 5.0);
        for nfft = [50 250 1000]
            x = DATA(1:min(end, 5*PARAMS.fs), 1);
            w = hanning(nfft);
            [pp,ff] = pwelch(x, w, 0, nfft, PARAMS.fs);
            db = 10*log10(pp);
            tr_dump(opt.out, sprintf('pwelch_nfft%d__db',nfft), db, 'binary');
            tr_dump(opt.out, sprintf('pwelch_nfft%d__f',nfft),  ff, 'binary');
            % what actually reaches disk: fwrite(...,'int8') rounds and saturates
            tr_dump(opt.out, sprintf('pwelch_nfft%d__int8',nfft), int8(db), 'binary');
            tr_dump(opt.out, sprintf('pwelch_nfft%d__meta',nfft), struct( ...
                'file',ref,'nfft',nfft,'noverlap',0,'fs',PARAMS.fs, ...
                'nsamples',numel(x),'window','hanning(nfft)', ...
                'source','calc_ltsa.m:159-162'));
            fprintf('  pwelch nfft=%4d -> %d bins\n', nfft, numel(db));
        end
        % --- int8 quantisation, tie-breaking and saturation, on chosen values
        % The pwelch dumps above never land on a .5 tie -- measured, the closest any
        % of them comes is 0.0054 -- so they cannot tell us how MATLAB breaks one.
        % That matters: MATLAB rounds half AWAY from zero while numpy's round() goes
        % half to EVEN, so 0.5, 2.5 and 126.5 disagree.  These probe values pin the
        % rule down where real data does not.  Saturation is included because an LTSA
        % value outside [-128,127] dB is clipped, not wrapped (ltsa.md 4).
        q8_probe = [-200.5 -128.5 -127.5 -1.5 -0.5 -0.4 0 0.4 0.5 1.5 2.5 ...
                    126.5 127.4 127.5 200.5]';
        tr_dump(opt.out,'int8_quant__in',  q8_probe, 'binary');
        tr_dump(opt.out,'int8_quant__out', int8(q8_probe), 'binary');
        tr_dump(opt.out,'int8_quant__meta', struct( ...
            'note','MATLAB int8() and fwrite(...,''int8'') round half away from zero and saturate', ...
            'source','calc_ltsa.m write path; see ltsa.md 4'));
        fprintf('  int8 quantisation probe -> %d values\n', numel(q8_probe));

        % the exact input samples, so Python starts from identical data
        load_segment(opt, ref, 0, 5.0);
        tr_dump(opt.out,'dsp_input_samples', DATA(:,1), 'binary');
    else
        warning('dump_reference:noRef','baseline fixture missing; skipping spectrogram dumps');
    end

    % --- transfer function interpolation (plot_specgram.m:64-77)
    tf_freq = [10 100 1000 10000 50000 100000];
    tf_uppc = [ 5  12   30    45    52     55];
    fq = (0:100:120000)';
    tr_dump(opt.out,'tf_interp__in',  [tf_freq(:) tf_uppc(:)], 'binary');
    tr_dump(opt.out,'tf_interp__f',   fq, 'binary');
    tr_dump(opt.out,'tf_interp__out', interp1(tf_freq,tf_uppc,fq,'linear','extrap'), 'binary');
    tr_dump(opt.out,'tf_interp__meta', struct('method','linear','extrap',true, ...
        'note','note deliberate extrapolation below 10 Hz and above 100 kHz', ...
        'source','plot_specgram.m:64-77'));

    % --- logfmap (log-frequency spectrogram axis).  M is dense flen x flen; two sizes are
    % enough to pin the mapping down without bloating the committed dump.
    for flen = [64 129]
        [M,N] = logfmap(flen,4,flen);
        tr_dump(opt.out, sprintf('logfmap_%d__M',flen), full(M), 'binary');
        tr_dump(opt.out, sprintf('logfmap_%d__N',flen), full(N), 'binary');
    end
end

%% ================================================================= ltsa
if want('ltsa')
    fprintf('--- ltsa ---\n');
    lt = dir(fullfile(opt.fixtures,'*.ltsa'));
    if isempty(lt)
        fprintf('  no .ltsa fixtures found -- run make_ltsa_fixture.m first, skipping\n');
    else
        tr_headless_handles();
        out = {};
        for k = 1:numel(lt)
            PARAMS = [];
            PARAMS.ltsa.inpath = opt.fixtures;
            PARAMS.ltsa.infile = lt(k).name;
            read_ltsahead;

            rec = struct('file',lt(k).name, ...
                         'ver',PARAMS.ltsa.ver, ...
                         'dirStartLoc',PARAMS.ltsa.dirStartLoc, ...
                         'dataStartLoc',PARAMS.ltsa.dataStartLoc, ...
                         'tave',PARAMS.ltsa.tave, ...
                         'dfreq',PARAMS.ltsa.dfreq, ...
                         'fs',PARAMS.ltsa.fs, ...
                         'nfft',PARAMS.ltsa.nfft, ...
                         'nf',PARAMS.ltsa.nf, ...
                         'nrftot',PARAMS.ltsa.nrftot, ...
                         'nxwav',PARAMS.ltsa.nxwav, ...
                         'ch',PARAMS.ltsa.ch, ...
                         'byteloc',PARAMS.ltsa.byteloc, ...
                         'nave',PARAMS.ltsa.nave, ...
                         'rfileid',PARAMS.ltsahd.rfileid, ...
                         'fnames',{cellstr(char(PARAMS.ltsahd.fname))});
            out{end+1} = rec; %#ok<AGROW>

            sn = safename(lt(k).name);
            tr_dump(opt.out, sprintf('ltsa_dnums__%s',sn), ...
                    [PARAMS.ltsa.dnumStart(:), PARAMS.ltsa.dnumEnd(:)], 'binary');

            % a data block, read exactly the way the plot path reads it
            PARAMS.ltsa.plot.dnum  = PARAMS.ltsa.start.dnum;
            PARAMS.ltsa.tseg.hr    = 1/60;      % 1 minute
            PARAMS.ltsa.tseg.sec   = 60;
            PARAMS.ltsa.save.dnum  = PARAMS.ltsa.start.dnum;
            read_ltsadata;
            tr_dump(opt.out, sprintf('ltsa_block__%s',sn), PARAMS.ltsa.pwr, 'binary');
            tr_dump(opt.out, sprintf('ltsa_block__%s__meta',sn), struct( ...
                'file',lt(k).name,'tseg_hr',PARAMS.ltsa.tseg.hr, ...
                'plotStartRawIndex',PARAMS.ltsa.plotStartRawIndex, ...
                'plotStartBin',PARAMS.ltsa.plotStartBin, ...
                'source','read_ltsadata.m'));
            fprintf('  %-40s ver=%d nrftot=%d nf=%d\n', lt(k).name, ...
                    PARAMS.ltsa.ver, PARAMS.ltsa.nrftot, PARAMS.ltsa.nf);
        end
        tr_dump(opt.out,'ltsa_headers',out);
    end
end

fprintf('\nDone.  Reference dump written to %s\n', opt.out);
fprintf('Commit fixtures/reference/ so the parity suite runs without MATLAB.\n');
end


%% ==================================================================== helpers

function cases = readseg_cases(fixdir, fname)
%READSEG_CASES  Choose interesting plot start times for one fixture.
%
% Offsets are seconds from the file's first raw-file start.  The interesting ones are:
% the very beginning, an offset inside raw file 1, an offset that lands a raw-file boundary
% in the middle of the window (the splice case), and a window starting inside raw file 2.

f = fopen(fullfile(fixdir,fname),'r');
fseek(f,22,'bof'); nch  = fread(f,1,'uint16');
fseek(f,34,'bof'); bits = fread(f,1,'uint16');
fseek(f,44,'bof'); wver = fread(f,1,'uint8');
fseek(f,80,'bof'); nrf  = fread(f,1,'uint16');
% the raw-file table starts after the harp chunk, which is longer in v2
if wver == 2, tbl = 100 + 4*nch; else, tbl = 100; end
fseek(f,tbl+12,'bof'); byte_length = fread(f,1,'uint32');
fseek(f,tbl+20,'bof'); fs          = fread(f,1,'uint32');
fclose(f);
raw_sec = byte_length / (nch * bits/8) / fs;

% keep each dumped block under ~25k samples/channel so fixtures/reference stays small
maxsec = 25000 / fs;

cases = struct('offset_sec',{},'tseg_sec',{},'label',{});
cases(end+1) = struct('offset_sec',0, ...
                      'tseg_sec',min([1.0, raw_sec/2, maxsec]), 'label','bof');
cases(end+1) = struct('offset_sec',raw_sec/4, ...
                      'tseg_sec',min([0.5, raw_sec/4, maxsec]), 'label','mid_raw1');
if nrf > 1
    % window straddling the raw-file 1 -> 2 boundary: the splice-and-delimit case
    tseg = min([0.5, maxsec]);
    cases(end+1) = struct('offset_sec',max(0,raw_sec - tseg/2), 'tseg_sec',tseg, ...
                          'label','across_boundary');
end
end


function rec = run_readseg(opt, fname, c)
%RUN_READSEG  Set up PARAMS the way initdata would, call readseg, dump DATA.
%
% initdata.m also touches ~15 GUI handles that are irrelevant to the byte arithmetic, so the
% handful of PARAMS fields it sets are replicated here instead of shimming all of them.
% Everything that affects the read itself comes from rdxwavhd and from readseg unmodified.

global PARAMS DATA %#ok<GVMIS>

tr_headless_handles();
PARAMS = [];

PARAMS.inpath = opt.fixtures;
PARAMS.infile = fname;
PARAMS.ftype  = 2;                 % xwav
rdxwavhd;

% --- replicating initdata.m for the non-GUI fields only
PARAMS.ch          = 1;
PARAMS.samp.head   = 0;
PARAMS.samp.null   = 0;
PARAMS.fmax        = PARAMS.fs/2;
PARAMS.freq0       = 0;
PARAMS.freq1       = PARAMS.fs/2;
PARAMS.start.dvec  = datevec(PARAMS.start.dnum);
PARAMS.plot.dvec   = PARAMS.start.dvec;
PARAMS.plot.dnum   = PARAMS.start.dnum;
PARAMS.save.dnum   = PARAMS.start.dnum;
PARAMS.plot.initbytel = PARAMS.xhd.byte_loc(1);
PARAMS.plot.bytelength = PARAMS.plot.initbytel;
PARAMS.tseg.step   = -1;
PARAMS.filter      = 0;
PARAMS.tf.flag     = 0;
% ---

PARAMS.tseg.sec  = c.tseg_sec;
PARAMS.plot.dnum = PARAMS.start.dnum + c.offset_sec/86400;
PARAMS.plot.dvec = datevec(PARAMS.plot.dnum);

readseg;      % the real thing: check_time + the byte arithmetic

tag = sprintf('readseg__%s__%s', safename(fname), c.label);
% DATA is double, but holds integer counts unless gain divided them; store int32 when that
% is lossless, purely to keep the committed reference dump small
if all(DATA(:) == fix(DATA(:))) && max(abs(DATA(:))) < 2^31
    tr_dump(opt.out, [tag '__data'], int32(DATA), 'binary');
else
    tr_dump(opt.out, [tag '__data'], DATA, 'binary');
end
tr_dump(opt.out, [tag '__meta'], struct( ...
    'file',fname, 'label',c.label, ...
    'requested_offset_sec',c.offset_sec, 'tseg_sec',c.tseg_sec, ...
    'tseg_samp',PARAMS.tseg.samp, ...
    'fs',PARAMS.fs, 'nch',PARAMS.nch, 'nBits',PARAMS.nBits, ...
    'xgain',PARAMS.xgain(1), ...
    'currentIndex',PARAMS.raw.currentIndex, ...
    'delimit_time',getfielddef(PARAMS.raw,'delimit_time',[]), ...
    'plot_dnum_hex',sprintf('%bx',PARAMS.plot.dnum), ...
    'data_size',size(DATA), ...
    'source','readseg.m + check_time.m'));

rec = struct('tag',tag,'file',fname,'label',c.label, ...
             'offset_sec',c.offset_sec,'tseg_sec',c.tseg_sec, ...
             'currentIndex',PARAMS.raw.currentIndex,'data_size',size(DATA));
fprintf('  %-44s %-16s -> %dx%d  idx=%s\n', fname, c.label, size(DATA,1), size(DATA,2), ...
        mat2str(PARAMS.raw.currentIndex));
end


function load_segment(opt, fname, offset_sec, tseg_sec)
%LOAD_SEGMENT  Populate PARAMS/DATA from a fixture, for the DSP dumps.
global PARAMS %#ok<GVMIS>
c = struct('offset_sec',offset_sec,'tseg_sec',tseg_sec,'label','dsp');
run_readseg_quiet(opt, fname, c);
end


function run_readseg_quiet(opt, fname, c)
global PARAMS DATA %#ok<GVMIS>
tr_headless_handles();
PARAMS = [];
PARAMS.inpath = opt.fixtures;  PARAMS.infile = fname;  PARAMS.ftype = 2;
rdxwavhd;
PARAMS.ch = 1; PARAMS.samp.head = 0; PARAMS.samp.null = 0;
PARAMS.fmax = PARAMS.fs/2; PARAMS.freq0 = 0; PARAMS.freq1 = PARAMS.fs/2;
PARAMS.start.dvec = datevec(PARAMS.start.dnum);
PARAMS.save.dnum = PARAMS.start.dnum;
PARAMS.plot.initbytel = PARAMS.xhd.byte_loc(1);
PARAMS.plot.bytelength = PARAMS.plot.initbytel;
PARAMS.tseg.step = -1; PARAMS.filter = 0; PARAMS.tf.flag = 0;
PARAMS.tseg.sec = c.tseg_sec;
PARAMS.plot.dnum = PARAMS.start.dnum + c.offset_sec/86400;
PARAMS.plot.dvec = datevec(PARAMS.plot.dnum);
readseg;
end


function name = pick_fixture(listing, stem)
name = '';
for k = 1:numel(listing)
    if startsWith(listing(k).name, stem); name = listing(k).name; return; end
end
end

function s = safename(f)
s = regexprep(f,'[^A-Za-z0-9]+','_');
end

function v = getfielddef(s, f, d)
if isstruct(s) && isfield(s,f); v = s.(f); else; v = d; end
end

function out = ternary(c,a,b)
if c; out = a; else; out = b; end
end
