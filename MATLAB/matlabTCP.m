while true
    t = tcpclient('localhost', 11101);
    if t.NumBytesAvailable > 0
        output = read(t);
        data = char(output(1:4));
        pwrcmd = str2double(data)
    end
end
clear t;