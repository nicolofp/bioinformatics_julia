# --- secrets loader: KEY=value lines from JULIA_ENV_FILE (default ~/Documents/secrets/julia.env) into ENV
let f = get(ENV, "JULIA_ENV_FILE", joinpath(homedir(), "Documents", "secrets", "julia.env"))
    if isfile(f)
        for l in eachline(f)
            s = strip(l)
            (isempty(s) || startswith(s, "#") || !occursin('=', s)) && continue
            k, v = split(s, '='; limit = 2)
            ENV[strip(k)] = strip(strip(v), ['"', '\''])
        end
    end
end

using AWS, AWSS3, Arrow, CSV
@service S3

AWS_ACCESS_KEY_ID = ENV["AWS_ACCESS_KEY_ID_2"]
AWS_SECRET_ACCESS_KEY = ENV["AWS_SECRET_ACCESS_KEY_2"]
AWS_DEFAULT_REGION = "us-east-2"
creds = AWSCredentials(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
const AWS_GLOBAL_CONFIG = Ref{AWS.AWSConfig}()
AWS_GLOBAL_CONFIG[] = AWS.global_aws_config(region=AWS_DEFAULT_REGION, creds=creds)

#=
data = S3.list_objects("envbran"; aws_config=AWS_GLOBAL_CONFIG[])

p = S3Path("s3://provanik/diamonds.csv") 
q = S3Path("s3://envbran/methylation/table1.arrow")
DBInterface.execute(con, """LOAD httpfs;""")

con = DBInterface.connect(DuckDB.DB, ":memory:")
dt = DataFrame(DBInterface.execute(con,
           """
           SELECT *
           FROM 's3://provanik/diamonds.csv'
           """));

df = DataFrame(Arrow.Table("https://envbran.s3.us-east-2.amazonaws.com/methylation/table1.arrow"))

bytes = AWSS3.read(S3Path("s3://envbran/methylation/table1.arrow"))
bytes = AWSS3.read(S3Path("s3://provanik/diamonds.csv"))

typeof(bytes)
# Vector{UInt8} (alias for Array{UInt8, 1})

df = DataFrame(Arrow.Table(bytes))
df = CSV.read(bytes, DataFrame)
tmp = CSV.read(p, DataFrame)
tmp1 = DataFrame(Arrow.Table(q))
=#