function tr_dump(outdir, name, data, varargin)
%TR_DUMP  Write one reference artefact for the Triton-Python parity suite.
%
%   tr_dump(OUTDIR, NAME, S)              writes NAME.json from struct/cell/array S
%   tr_dump(OUTDIR, NAME, A, 'binary')    writes NAME.bin (raw little-endian) + NAME.json
%                                         sidecar giving shape and dtype
%
% Binary is required for anything floating point: JSON round-tripping loses bits, and the
% whole point of the parity suite is bit-exact comparison.  MATLAB writes column-major; the
% Python side reads with order='F' (see tests/conftest.py:load_reference_bin).
%
% Part of the Triton-Python Phase 0 parity tooling.

binary = any(strcmpi(varargin,'binary'));

if ~exist(outdir,'dir'); mkdir(outdir); end

if binary
    cls = class(data);
    switch cls
        case 'double', dtype = 'float64';
        case 'single', dtype = 'float32';
        case 'int8',   dtype = 'int8';
        case 'int16',  dtype = 'int16';
        case 'int32',  dtype = 'int32';
        case 'uint8',  dtype = 'uint8';
        otherwise
            error('tr_dump:unsupported','unsupported class %s for binary dump', cls);
    end
    fid = fopen(fullfile(outdir,[name '.bin']),'w','ieee-le');
    fwrite(fid, data, cls);
    fclose(fid);

    meta = struct('name', name, 'dtype', dtype, 'shape', size(data), ...
                  'order', 'F', 'matlab_class', cls);
    tr_writejson(fullfile(outdir,[name '.json']), meta);
else
    tr_writejson(fullfile(outdir,[name '.json']), data);
end
end


function tr_writejson(path, data)
try
    txt = jsonencode(data, 'PrettyPrint', true);   % R2021a+
catch
    txt = jsonencode(data);
end
fid = fopen(path,'w');
fprintf(fid,'%s\n',txt);
fclose(fid);
end
