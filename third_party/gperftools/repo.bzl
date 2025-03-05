#load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
#
#def _gperftools_extension(ctx):
#    for mod in ctx.modules:
#        http_archive(
#            name = "com_github_gperftools_gperftools",
#            url = "https://github.com/gperftools/gperftools/archive/a81b2ebbc2cec046aed5d571cdc783c49b48843a.tar.gz",
#            sha256 = "43eb268bcc53a8742b400a58a8a30b2c9cd9e61b2733754f5f23aa451ad32dc9",
#            strip_prefix = "gperftools-a81b2ebbc2cec046aed5d571cdc783c49b48843a",
#            build_file = "//third_party/gperftools:gperftools.BUILD",
#        )
#
#gperftools_ext = module_extension(implementation = _gperftools_extension)
