# A minimal JSON reader for the tests (no packages): objects → Dict{String,Any}, arrays →
# Vector{Any}, numbers → Int or Float64 (Float64 parsed correctly rounded), null → nothing.

function parse_json(text::AbstractString)
    s = codeunits(String(text))
    i = Ref(1)
    ws() = while i[] <= length(s) && s[i[]] in (0x20, 0x09, 0x0a, 0x0d)
        i[] += 1
    end
    function value()
        ws()
        c = Char(s[i[]])
        if c == '{'
            i[] += 1
            d = Dict{String,Any}()
            ws()
            if Char(s[i[]]) == '}'
                i[] += 1
                return d
            end
            while true
                ws()
                k = str()
                ws()
                Char(s[i[]]) == ':' || error("JSON: ':' expected at byte $(i[])")
                i[] += 1
                d[k] = value()
                ws()
                c = Char(s[i[]])
                i[] += 1
                c == '}' && return d
                c == ',' || error("JSON: ',' expected at byte $(i[] - 1)")
            end
        elseif c == '['
            i[] += 1
            a = Any[]
            ws()
            if Char(s[i[]]) == ']'
                i[] += 1
                return a
            end
            while true
                push!(a, value())
                ws()
                c = Char(s[i[]])
                i[] += 1
                c == ']' && return a
                c == ',' || error("JSON: ',' expected at byte $(i[] - 1)")
            end
        elseif c == '"'
            return str()
        elseif c == 't'
            i[] += 4
            return true
        elseif c == 'f'
            i[] += 5
            return false
        elseif c == 'n'
            i[] += 4
            return nothing
        else
            j = i[]
            while i[] <= length(s) && Char(s[i[]]) in "+-0123456789.eE"
                i[] += 1
            end
            t = String(s[j:i[]-1])
            return occursin(r"[.eE]", t) ? parse(Float64, t) : parse(Int, t)
        end
    end
    function str()
        Char(s[i[]]) == '"' || error("JSON: string expected at byte $(i[])")
        i[] += 1
        buf = IOBuffer()
        while true
            b = s[i[]]
            if b == UInt8('"')
                i[] += 1
                return String(take!(buf))
            elseif b == UInt8('\\')
                e = Char(s[i[]+1])
                if e == 'u'
                    write(buf, Char(parse(UInt16, String(s[i[]+2:i[]+5]), base = 16)))
                    i[] += 6
                else
                    write(buf, get(Dict('n' => '\n', 't' => '\t', 'r' => '\r', 'b' => '\b', 'f' => '\f'), e, e))
                    i[] += 2
                end
            else
                write(buf, b)
                i[] += 1
            end
        end
    end
    return value()
end

read_json(path) = parse_json(read(path, String))
