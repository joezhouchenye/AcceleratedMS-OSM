function data_complex = read_complex_bin(filename)

    % 以二进制方式打开文件
    fid = fopen(filename, 'rb');
    if fid == -1
        error('无法打开文件: %s', filename);
    end

    % 按 uint32 读取，每个 uint32 对应一个复数
    raw = fread(fid, inf, 'uint32=>uint32');
    fclose(fid);

    % 低16位：实部
    real_part = bitand(raw, uint32(65535));

    % 高16位：虚部
    imag_part = bitshift(raw, -16);

    % 转成 double，方便后续运算
    real_part = double(real_part);
    imag_part = double(imag_part);

    % 非零减32768，否则保持0
    real_part(real_part ~= 0) = real_part(real_part ~= 0) - 32768;
    imag_part(imag_part ~= 0) = imag_part(imag_part ~= 0) - 32768;

    % 组成复数
    data_complex = complex(real_part, imag_part);

end