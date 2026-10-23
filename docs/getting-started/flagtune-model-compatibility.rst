FlagTune model compatibility and publication
===========================================

Explicit candidate configurations
---------------------------------

``params`` declares each parameter's legal values. Without ``configs``, the
candidate domain is their Cartesian product. An optional ``configs`` list
restricts it to the complete combinations explicitly listed, preserving linked
parameters such as ``PIPE_STAGES == num_stages``. Mapping-key order is ignored
and normalized to parameter declaration order; candidate-list order is retained.
Every candidate must contain exactly the declared parameters with legal values.

The model schema remains ``format_version: 5``. This is an optional extension
requiring FlagTune runtime **1.1.0 or later**, independently of the model package
version. New explicit-config exports declare ``flagtune_version_min: 1.1.0``.
Previously published, digest-bound bundles without that declaration are also
recognized as requiring 1.1.0 when they contain ``configs``; their bytes must not
be rewritten. The runtime checks requirements before interpreting schema fields,
including during validation of every child in a platform package.

Runtime 1.1.0 continues to read version-5 models without ``configs``. Runtime
1.0.0 cannot read explicit-config models and must never ignore the field: that
would expand the candidate space into untrained combinations. A single such
child can prevent 1.0.0 from reading the whole package, including its ``mul``
models. Version checks added to newer code cannot retrofit the old parser.

Manifest isolation and release order
-----------------------------------

The default endpoint for runtime 1.1.0 is
``https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtune-xgb-manifest-v1.1.0.tar.gz``.
Keep the legacy ``flagtune-xgb-manifest-v1.0.0.tar.gz`` and unversioned Manifest
compatible with 1.0.0: do not expose explicit-config packages there. Existing
clients compute the highest package SemVer and do not filter by runtime version.
An explicitly configured local Manifest or URL remains authoritative.

Release the T-Head MM/MV integration in this order:

1. Publish the immutable ``flagtune-xgb-thead-zw810e_v1.1.0.tar.gz`` archive at
   its catalog URL. Verify SHA-256
   ``0a089ab5cba39867a0ce57992a65e7991e33ca26549dcf893e736bcff3feb0cc``.
2. Stage the 1.1.0 runtime and a new Manifest generated from ``PACKAGE_CATALOG``.
   Load every child, run real XGBoost predictions, verify candidate membership,
   and execute operator correctness and AUTO fallback tests.
3. Publish the versioned 1.1.0 Manifest and verify the production endpoint.
   Release FlagTree with runtime 1.1.0 only when this endpoint is available.
4. Release the companion FlagGems MM/MV bindings with an explicit requirement
   for that FlagTree release. FlagGems' bundled Manifest must obey the same
   runtime requirement. Users retaining the old runtime must also retain a
   compatible FlagGems release and pin the legacy Manifest/package.

Do not replace an existing same-version model archive. Updating the source
catalog alone does not publish either a model or a hosted Manifest. Keep the
generated Manifest bundle outside the source tree, for example::

    python -m triton.flagtune.training.manifest_generator \
        --bundle-output /path/to/release/flagtune-xgb-manifest-v1.1.0.tar.gz

Validation
----------

The opt-in remote integration test defaults to the same endpoint as production::

    FLAGTUNE_RUN_REMOTE_INTEGRATION=1 \
      python -m pytest python/test/flagtune/test_model_remote_integration.py -v

It downloads and validates the complete package, loads the real XGBoost
predictors and feature schemas, predicts over legal candidates, and repeats
loading from the disk cache with networking disabled. All ten T-Head 1.1.0
children are covered. A staging endpoint can be selected explicitly with
``FLAGTUNE_TEST_MANIFEST_URL``; staging results are not evidence that the
production endpoint has been published.
``FLAGTUNE_TEST_LOCAL_MANIFEST`` can select a generated local Manifest for
pre-publication testing of the published archive URLs and their SHA-256 hashes.

Existing ``mul`` packages for NVIDIA, MetaX, MThreads, Hygon and T-Head, and
NVIDIA/MetaX ``mm`` models, must also be tested with the updated runtime. Model
loading and prediction are CPU tests; kernel execution requires each respective
device and a compatible FlagGems checkout. The companion FlagGems script
``tools/flagtune_published_regression.py`` checks real proposals, numerical
correctness and, with ``--scenario unavailable``, AUTO fallback. Use separate
fresh model and kernel caches for each scenario and retain source/device details
with the results. Mark missing hardware explicitly as unverified.

The PPU compatibility baseline is SDK 2.1. Execute the ordinary memory/inline
move and FP16/BF16 MMA tests on an idle PPU with::

    FLAGTUNE_RUN_PPU_SDK_TESTS=1 \
      python -m pytest third_party/ppu/python/test/unit/language/test_sdk21.py -v

Cost Model failures handled by FlagGems AUTO select the existing Default or
Expanded fallback. Explicit REQUIRED mode reports the failure instead. A
fallback success must not be counted as successful Cost Model inference.
