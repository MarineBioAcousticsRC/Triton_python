function make_ltsa_fixture(varargin)
%MAKE_LTSA_FIXTURE  Build .ltsa fixtures from the generated x.wav corpus, using Triton itself.
%
% The LTSA files in the fixture corpus are produced by the real MATLAB pipeline
% (get_headers -> ck_ltsaparams -> write_ltsahead -> calc_ltsa), not by a Python writer.
% That is the point: the Phase 1 milestone is a Python mkltsa whose output is byte-identical
% to these.
%
% Usage
%   make_ltsa_fixture
%   make_ltsa_fixture('fixtures', D, 'tave', 1, 'dfreq', 100)
%
% Options
%   'fixtures'  fixture directory   (default: <repo>\fixtures\generated)
%   'triton'    Triton-master path  (default: from `which triton`)
%   'tave'      seconds per LTSA time bin   (default 1)
%   'dfreq'     Hz per frequency bin        (default 100)
%   'files'     cellstr of x.wav basenames to include (default: the 1-channel, honest-fs ones)
%
% ONE MANUAL STEP
%   write_ltsahead.m line 18 reads
%       if ~exist('PARAMS.ltsa.outfile','var')
%   `exist` cannot test a dotted field name, so this is always true and uiputfile always
%   opens -- even though this script has already set the output name.  A Save dialog will
%   therefore appear once per LTSA; accept the suggested filename printed in the console.
%
%   Worth fixing upstream in Triton-master regardless of this port:
%       if ~isfield(PARAMS.ltsa,'outfile') || isempty(PARAMS.ltsa.outfile)
%
% Part of the Triton-Python Phase 0 parity tooling.

here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));

p = inputParser;
addParameter(p,'fixtures',fullfile(repo,'fixtures','generated'));
addParameter(p,'triton','');
addParameter(p,'tave',1);
addParameter(p,'dfreq',100);
addParameter(p,'files',{});
parse(p,varargin{:});
opt = p.Results;

if isempty(opt.triton)
    w = which('triton');
    if isempty(w)
        error('make_ltsa_fixture:noTriton','add Triton-master to the path, or pass ''triton''');
    end
    opt.triton = fileparts(w);
end
addpath(opt.triton);
addpath(here);

global PARAMS %#ok<GVMIS>
tr_headless_handles();

% Default set: single-channel fixtures whose fmt-chunk rate matches the raw-file table.
% ck_ltsaparams aborts when they disagree (see docs/formats/xwav.md 5.1), which is itself
% worth a parity test but cannot produce an LTSA.
if isempty(opt.files)
    opt.files = { ...
        'xwav_v1_cont_1ch_16b_10k.x.wav', ...
        'xwav_v1_duty_1ch_16b_10k.x.wav', ...
        'xwav_v1_cont_1ch_16b_200k.x.wav'};
end

for k = 1:numel(opt.files)
    fname = opt.files{k};
    src   = fullfile(opt.fixtures, fname);
    if exist(src,'file') ~= 2
        warning('make_ltsa_fixture:missing','skipping absent fixture %s', fname);
        continue
    end

    [~, stem] = fileparts(fname);            % still ends in '.x'
    stem = regexprep(stem,'\.x$','');
    outname = sprintf('%s__tave%g_df%g.ltsa', stem, opt.tave, opt.dfreq);

    fprintf('\n=== %s -> %s ===\n', fname, outname);
    fprintf('    when the Save dialog opens, save as: %s\n', ...
            fullfile(opt.fixtures, outname));

    PARAMS = [];
    init_ltsaparams;                          % defaults

    % --- what get_ltsadir would have asked for interactively
    PARAMS.ltsa.indir  = [opt.fixtures filesep];
    PARAMS.ltsa.fname  = char(fname);
    PARAMS.ltsa.ftype  = 2;                   % xwav
    PARAMS.ltsa.gen    = 1;

    % --- what get_ltsaparams would have asked for interactively
    PARAMS.ltsa.tave   = opt.tave;
    PARAMS.ltsa.dfreq  = opt.dfreq;
    PARAMS.ltsa.dtype  = 1;                   % HARP sector geometry
    PARAMS.ltsa.ch     = 1;

    PARAMS.ltsa.outdir  = [opt.fixtures filesep];
    PARAMS.ltsa.outfile = outname;            % ignored by the guard bug above

    get_headers;
    if PARAMS.ltsa.gen == 0
        warning('make_ltsa_fixture:headers','get_headers failed for %s', fname); continue
    end
    ck_ltsaparams;
    write_ltsahead;
    if PARAMS.ltsa.gen == 0
        warning('make_ltsa_fixture:cancelled','cancelled at the save dialog'); continue
    end
    calc_ltsa;

    made = fullfile(PARAMS.ltsa.outdir, PARAMS.ltsa.outfile);
    d = dir(made);
    fprintf('    wrote %s (%d bytes, ver %d, nrftot %d, nfft %d)\n', ...
            PARAMS.ltsa.outfile, d.bytes, PARAMS.ltsa.ver, PARAMS.ltsa.nrftot, ...
            PARAMS.ltsa.nfft);
end

fprintf('\nNow run dump_reference(''sections'',{''ltsa''}) to capture the read-side truth.\n');
end
