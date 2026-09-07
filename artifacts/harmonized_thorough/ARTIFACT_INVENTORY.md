# ARTIFACT INVENTORY: HARMONIZED MULTI-SEED MATRIX (READ-ONLY AUDIT)

**Audit Timestamp**: 2026-09-07 22:47:37  
**Repository Root**: `d:\cloud-classification-cnn-mobile`  
**Benchmark Output Root**: `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough`  
**Audit Scope**: Complete 36-run harmonized benchmark ($3\text{ architectures} \times 4\text{ arms} \times 3\text{ seeds} = 36\text{ runs}$)  
**Audit Mode**: Strictly READ-ONLY (no execution, re-training, synthesis, recompilation, modification, git commit, or git push)  

---
## Executive Summary of Artifact Inventory

All **36 production benchmark runs** have been physically verified on disk under `artifacts/harmonized_thorough/`. Every single run directory contains exactly the expected five canonical files: the run summary JSON, the min-val-loss model checkpoint (`.pth`), the max-val-macro-F1 checkpoint (`_best_f1.pth`), and their respective provenance metadata sidecars (`.meta.json`).

Key physical inventory findings:
1. **Physical Completeness**: 36 of 36 run directories exist; 100% of summary JSONs (36/36), min-loss checkpoints (36/36), max-F1 checkpoints (36/36), and metadata sidecars (72/72) are present and verified by SHA-256 cryptographic hashes.
2. **Provenance & Seed Integrity**: Recorded seeds match intended seeds across all 36 runs (100% match). Recorded epoch completions match configured epoch targets across all 36 runs (100% match). All checkpoints were newly trained from scratch (`checkpoint_reused: false`).
3. **Scientific Reconciliation**: Every single metric previously reported (top-1 accuracy, balanced accuracy, macro-F1 across both min-loss and max-F1 criteria) reconciles with 100.00% precision against the summary JSONs on disk, with **0 discrepancies across all 378 verified metric points**.
4. **Missing Artifact Disclosures**: Per-sample predictions (`y_true`, `y_pred`, `y_prob`) and machine-readable confusion matrices (`.npy`, `.csv`, `.json`) are **ABSENT** from all run directories (they were never written to disk by `run_harmonized.py`). Static PNG confusion plots were written to a single shared repository directory (`artifacts/figures/`) without per-seed namespacing, resulting in successive run overwrites. These can only be reproduced by re-evaluating the physical `.pth` checkpoints.

---

## 1. Per-Run Evidence Table (All 36 Benchmark Runs)

The tables below detail the physical artifact presence, 64-hex SHA-256 checksums, byte sizes, filesystem timestamps, and execution provenance for every single run across the matrix ($3\text{ architectures} \times 4\text{ arms} \times 3\text{ seeds} = 36\text{ runs}$).

### 1.1 Summary / Results JSON Physical Evidence (36 Runs)

| # | Arch | Arm | Seed | Output Directory Path | Summary JSON File | Exists | Byte Size | SHA-256 Checksum | File mtime |
| :-: | :--- | :--- | :-: | :--- | :--- | :-: | :-: | :--- | :--- |
| 1 | resnet18 | ccsn15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn15_seed42` | `harmonized_summary_resnet18.json` | **Yes** | 16,353 | `fcd586cc2a2fc8051e75116035a9fbcd7a0c0409b586be0e0f639ed0f538b545` | 2026-09-07 17:50:08 |
| 2 | resnet18 | ccsn15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn15_seed43` | `harmonized_summary_resnet18.json` | **Yes** | 16,357 | `b51d631049130ba253cb6baa2315c6fa336d03dfafcb46f49e6b4d88e956ec9e` | 2026-09-07 17:52:10 |
| 3 | resnet18 | ccsn15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn15_seed44` | `harmonized_summary_resnet18.json` | **Yes** | 16,357 | `2eca0f510958e221e53a7fb7c3e9acae29ea4887e62acc2b41dcda2a2fe5331e` | 2026-09-07 17:53:33 |
| 4 | resnet18 | ccsn90 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn90_seed42` | `harmonized_summary_resnet18.json` | **Yes** | 16,356 | `335cc1e6f9c2ff49d46222e2ef24f5cd266cfb630d3b7d09156cbbe48945b630` | 2026-09-07 17:57:52 |
| 5 | resnet18 | ccsn90 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn90_seed43` | `harmonized_summary_resnet18.json` | **Yes** | 16,354 | `718a4bf40fa0222f0cf3b9e167ee5987dcda669e410fd0a16b4d6bcf85a3a738` | 2026-09-07 18:01:43 |
| 6 | resnet18 | ccsn90 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\ccsn90_seed44` | `harmonized_summary_resnet18.json` | **Yes** | 16,349 | `8ba565ef0c084070d8a7f9b122a47809709d58ca5696a08035ed16af11c4aa86` | 2026-09-07 18:05:34 |
| 7 | resnet18 | gcd15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\gcd15_seed42` | `harmonized_summary_resnet18.json` | **Yes** | 16,347 | `51c9b7e9f328ae3535bfdd9813368542f4925e10c2f7598ab70f9bb515f20067` | 2026-09-07 18:09:37 |
| 8 | resnet18 | gcd15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\gcd15_seed43` | `harmonized_summary_resnet18.json` | **Yes** | 16,338 | `2c2f77a2700c060fa7c3e039ab15a43e25b65d035d72ee935b68bdb7082f0fd1` | 2026-09-07 18:13:41 |
| 9 | resnet18 | gcd15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\gcd15_seed44` | `harmonized_summary_resnet18.json` | **Yes** | 16,335 | `b6cd728c1579bdd5aab49233a0da73e120fe72a9e8bb6c647aaf9cf6691692a7` | 2026-09-07 18:17:41 |
| 10 | resnet18 | joint15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\joint_seed42` | `harmonized_summary_resnet18.json` | **Yes** | 22,648 | `45fc537dec746da1c4bb57aa9fcd5fdbd7d722dae02aabc28722887c65546298` | 2026-09-07 18:22:35 |
| 11 | resnet18 | joint15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\joint_seed43` | `harmonized_summary_resnet18.json` | **Yes** | 22,648 | `4ffa3452d1c572d370f2ec115319952c36ede73bd95eedd7bd4dc9cb1a7bc521` | 2026-09-07 18:27:31 |
| 12 | resnet18 | joint15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet18\joint_seed44` | `harmonized_summary_resnet18.json` | **Yes** | 22,645 | `3c89136c611a6893c018f717086e95b4c7515e5a59738042d661099ba4ed33ac` | 2026-09-07 18:32:22 |
| 13 | resnet34 | ccsn15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn15_seed42` | `harmonized_summary_resnet34.json` | **Yes** | 16,347 | `8c0c18527c246cbef420e633aee2dde72a2ac410b1b2b2b66bf76b4e7c38586c` | 2026-09-07 19:07:00 |
| 14 | resnet34 | ccsn15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn15_seed43` | `harmonized_summary_resnet34.json` | **Yes** | 16,341 | `7ea3455047c8e31845f772df430e2fdb7c288bb517db40cc09f21e6e6cfa8de0` | 2026-09-07 19:08:45 |
| 15 | resnet34 | ccsn15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn15_seed44` | `harmonized_summary_resnet34.json` | **Yes** | 16,342 | `a711b4f5333ffa6b1fd6f7c4a6e330902c3f57f505f656dccb813da23edcb12b` | 2026-09-07 19:10:31 |
| 16 | resnet34 | ccsn90 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn90_seed42` | `harmonized_summary_resnet34.json` | **Yes** | 16,348 | `eed2ccea693001e15fd76ddc3b950168996c13039a3df080ffd0638784c75e7d` | 2026-09-07 19:15:43 |
| 17 | resnet34 | ccsn90 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn90_seed43` | `harmonized_summary_resnet34.json` | **Yes** | 16,348 | `d35f30a9a10b6b9d23b4547fb0261e9c5eefdb0f90ebf833d928764b5924a0d3` | 2026-09-07 19:21:33 |
| 18 | resnet34 | ccsn90 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\ccsn90_seed44` | `harmonized_summary_resnet34.json` | **Yes** | 16,339 | `12a4149c2b29a9f2688696f98cdd5b71f0498c52bc4c82ea0e28ab7156fd42e1` | 2026-09-07 19:27:11 |
| 19 | resnet34 | gcd15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\gcd15_seed42` | `harmonized_summary_resnet34.json` | **Yes** | 16,336 | `45391de37da66c37715920eb2ef5a05095c63237808abeea2e232b3d7a39d9f1` | 2026-09-07 19:57:53 |
| 20 | resnet34 | gcd15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\gcd15_seed43` | `harmonized_summary_resnet34.json` | **Yes** | 16,327 | `c87063a43bb9cd7e702bfa9081b501fa8a7c8190c9f499c1810b27cfa125af9b` | 2026-09-07 20:11:09 |
| 21 | resnet34 | gcd15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\gcd15_seed44` | `harmonized_summary_resnet34.json` | **Yes** | 16,337 | `d7f9b9cb0b142d4d3ed4cfa7952e7e82e3b28d8efd7a41d5febef9c0be59ed5d` | 2026-09-07 20:21:16 |
| 22 | resnet34 | joint15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\joint_seed42` | `harmonized_summary_resnet34.json` | **Yes** | 22,660 | `f36155b0fac85646643a484819595499f44dd066b6a7d35dd1988edcafe26baf` | 2026-09-07 20:29:28 |
| 23 | resnet34 | joint15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\joint_seed43` | `harmonized_summary_resnet34.json` | **Yes** | 22,653 | `292e2b7cf5955aa81d1066cfaf6467222d219da914be56df70b3fb2c2e4f5eb0` | 2026-09-07 20:37:43 |
| 24 | resnet34 | joint15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet34\joint_seed44` | `harmonized_summary_resnet34.json` | **Yes** | 22,656 | `fe609bd81c9938ad3d28c01bd33f0659626ff1a990c6b067c353652d18a4592c` | 2026-09-07 20:45:53 |
| 25 | resnet50 | ccsn15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn15_seed42` | `harmonized_summary_resnet50.json` | **Yes** | 16,340 | `45fc037b2b5b74412bfe81f970f3ca228f8d9fb3b14f0a6ea0de7bef86914dbc` | 2026-09-07 20:47:02 |
| 26 | resnet50 | ccsn15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn15_seed43` | `harmonized_summary_resnet50.json` | **Yes** | 16,353 | `5f9ea8cdc93ab11c3a06881a1eb41e57029f095a856a87169094ba26a2cda4b3` | 2026-09-07 20:48:11 |
| 27 | resnet50 | ccsn15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn15_seed44` | `harmonized_summary_resnet50.json` | **Yes** | 16,336 | `672783c4d55f0ec1e786343c3fda15b69ff0f1b0f634e9921087f3565d8d0be5` | 2026-09-07 20:49:20 |
| 28 | resnet50 | ccsn90 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn90_seed42` | `harmonized_summary_resnet50.json` | **Yes** | 16,345 | `f6873aa1f0d37e77a52d4f5eba652dd963e2cd90110615f2e36ea0ea98ec4f1a` | 2026-09-07 20:54:19 |
| 29 | resnet50 | ccsn90 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn90_seed43` | `harmonized_summary_resnet50.json` | **Yes** | 16,356 | `51cc46cccfa06603e87af05d728b6b72b857d9aa19e03c54f362ae05e212b629` | 2026-09-07 21:00:45 |
| 30 | resnet50 | ccsn90 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\ccsn90_seed44` | `harmonized_summary_resnet50.json` | **Yes** | 16,352 | `5246196381a97e64a9120736bf6c3953dfbd7cd09d1f51a4d1bb5cc533e35ec5` | 2026-09-07 21:11:40 |
| 31 | resnet50 | gcd15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\gcd15_seed42` | `harmonized_summary_resnet50.json` | **Yes** | 16,332 | `5773f862e574316cd3c1dde274edc8024c122a930bf654145f1c241ba5ae0abc` | 2026-09-07 21:20:37 |
| 32 | resnet50 | gcd15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\gcd15_seed43` | `harmonized_summary_resnet50.json` | **Yes** | 16,335 | `e820306aee56fe34e0cd51108451c1133c4954f25bf0cc4e3b0f23b5817e335f` | 2026-09-07 21:29:08 |
| 33 | resnet50 | gcd15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\gcd15_seed44` | `harmonized_summary_resnet50.json` | **Yes** | 16,336 | `4d1fc688696936113a7ca991ad78345b35f3240412b5ebf8151b664b01dc2d28` | 2026-09-07 21:37:38 |
| 34 | resnet50 | joint15 | 42 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\joint_seed42` | `harmonized_summary_resnet50.json` | **Yes** | 22,654 | `03223fdaa3151d34405443e87cf3a4176de525a0cca188942ca2c47b017c10f1` | 2026-09-07 21:47:38 |
| 35 | resnet50 | joint15 | 43 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\joint_seed43` | `harmonized_summary_resnet50.json` | **Yes** | 22,655 | `d16e1f52b394527be3a09dc092e3d583762b28e415367761ba2dbedeb664a8da` | 2026-09-07 21:57:38 |
| 36 | resnet50 | joint15 | 44 | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\resnet50\joint_seed44` | `harmonized_summary_resnet50.json` | **Yes** | 22,653 | `b82793b28317aa9edc1ff280c911016da7819697ab842e61f66348f1af9aa421` | 2026-09-07 22:07:39 |

### 1.2 Model Checkpoints Physical Evidence: Min-Val-Loss & Max-Val-Macro-F1 (36 Runs)

| # | Arch | Arm | Seed | Min-Val-Loss Checkpoint (.pth) | Size (Bytes) | Min-Loss Checkpoint SHA-256 | Max-Val-F1 Checkpoint (_best_f1.pth) | Size (Bytes) | Max-F1 Checkpoint SHA-256 | Checkpoint mtime |
| :-: | :--- | :--- | :-: | :--- | :-: | :--- | :--- | :-: | :--- | :--- |
| 1 | resnet18 | ccsn15 | 42 | `ccsn_model_resnet18.pth` | 45,323,308 | `8662a5dbe34f3a898531d681fe1f6421f9c9315d60f35a4315d6ee10a0705813` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `d9bdade3efd9bc58664e15953c092d8cb2d032fe72e4b8e453bdfd5e56ba03a2` | 2026-09-07 17:49:45 |
| 2 | resnet18 | ccsn15 | 43 | `ccsn_model_resnet18.pth` | 45,323,308 | `d31d6f5822ff5c6c5a33036d93b1614655d1b92a4a02c8e77d94a46f1cb89d46` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `ce2907a659c1009d9e2cd1ebc0ead960684ce3046b99b90b3818f10e0bdc323c` | 2026-09-07 17:51:47 |
| 3 | resnet18 | ccsn15 | 44 | `ccsn_model_resnet18.pth` | 45,323,308 | `53efd23a6001f4690155c2625b13c7977a840c00c771556d67c372cb82086d56` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `c80146b2d2e6bc3b3a9b0daf5503aecb01dc88c039dbfe3859511824b4d9860b` | 2026-09-07 17:53:10 |
| 4 | resnet18 | ccsn90 | 42 | `ccsn_model_resnet18.pth` | 45,323,308 | `3f8b0fec5303af3fd7186ffc71fc3fe5fb60907abe4b3cc7dfc38166ab382bd4` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `8fe78c93d9747e4851a9bcf31205f51675173f22b07f9b42fe566ecc881a605e` | 2026-09-07 17:57:29 |
| 5 | resnet18 | ccsn90 | 43 | `ccsn_model_resnet18.pth` | 45,323,308 | `b6b15063b77ba37d2c19539a9f4cf89de5300ca97775daf8a3e3379d07507ea1` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `d94bf3e25f167b2965a5aa9e447c21732193311a89b63765aac2f99171ed5c83` | 2026-09-07 18:01:20 |
| 6 | resnet18 | ccsn90 | 44 | `ccsn_model_resnet18.pth` | 45,323,308 | `bbbdaa68d81a36057519bad063e4b7187a5d7b4bc4120d9acf145e076e26c140` | `ccsn_model_resnet18_best_f1.pth` | 45,324,436 | `165409140d9b9458796e8265a99bf6d0d5e2780be8ffa186af624389f576ab14` | 2026-09-07 18:05:11 |
| 7 | resnet18 | gcd15 | 42 | `gcd_model_resnet18.pth` | 45,323,175 | `8be3453b34c972511253ba7e06c289d59a60ccaecd263403fcb3868efbd45bc9` | `gcd_model_resnet18_best_f1.pth` | 45,324,303 | `ac815b49efd5a33295f2654b013304ae122d55a3d10643ea7f54dedf46c73e25` | 2026-09-07 18:09:14 |
| 8 | resnet18 | gcd15 | 43 | `gcd_model_resnet18.pth` | 45,323,175 | `9c69a9cd330ae8872910ebc8b7a12b5472128c564514c237c96b819eeec18416` | `gcd_model_resnet18_best_f1.pth` | 45,324,303 | `e8381d0f683705e0b0f98a13b73b20d04a15ef643cb34053b50266c22fc41ca7` | 2026-09-07 18:13:18 |
| 9 | resnet18 | gcd15 | 44 | `gcd_model_resnet18.pth` | 45,323,175 | `935ee55ad7e2b3d920c1876103a4b0b0a74174a554dc2e42d2930462872576d0` | `gcd_model_resnet18_best_f1.pth` | 45,324,303 | `969095790b39df128e09fc9045b5eb2b87bdc4135c481e01dc5b585559bb4381` | 2026-09-07 18:17:18 |
| 10 | resnet18 | joint15 | 42 | `harmonized_joint_resnet18.pth` | 45,324,170 | `4beeac7b4a70c8896d86c21bade82872f95790fcc150284b16666940dfa35b9c` | `harmonized_joint_resnet18_best_f1.pth` | 45,325,234 | `2d78a39ddb4aa57e82dbfa669d1a1d02a7718b092551f7d59ad726f0f462261d` | 2026-09-07 18:21:58 |
| 11 | resnet18 | joint15 | 43 | `harmonized_joint_resnet18.pth` | 45,324,170 | `66d4c9414f4d0259bfe9327e06fe5917973e2fb93678ea784cc9915a22af45b6` | `harmonized_joint_resnet18_best_f1.pth` | 45,325,234 | `684f7974eaef53903ff8d2b13fc0fab94fa1e88ad43ca89ea80bfcc12c0b26be` | 2026-09-07 18:26:55 |
| 12 | resnet18 | joint15 | 44 | `harmonized_joint_resnet18.pth` | 45,324,170 | `71ffd07428a407263d03ba38aae487143c7f1cb067e96a45c9f13154ef0935ce` | `harmonized_joint_resnet18_best_f1.pth` | 45,325,234 | `e71a18e5ffb440ea2e488fb75a20de1646320219b6a340455d17e07ad4cd55c4` | 2026-09-07 18:31:45 |
| 13 | resnet34 | ccsn15 | 42 | `ccsn_model_resnet34.pth` | 85,818,348 | `d7cb7f64fb5e865fa728166396fe285b4b4c9c5bf872d8c6b6a9ec04fda70a83` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `045c9051fc9596d2c3aea849c8c857a32f812a982bd476bac4ddc21d2aca6ed6` | 2026-09-07 19:06:28 |
| 14 | resnet34 | ccsn15 | 43 | `ccsn_model_resnet34.pth` | 85,818,348 | `fc138b8037896b4e026938b4366a5b6f3615f95e796fa10b74d10362e5988c2a` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `2ffb212a89fb0b4deaf3d0d2b07ab0c3e68b6a18190c267c3ccd3b15340ab92d` | 2026-09-07 19:08:13 |
| 15 | resnet34 | ccsn15 | 44 | `ccsn_model_resnet34.pth` | 85,818,348 | `337e09cb7ea5272b6b6f4e87e44a2b86f508538fdfe57b1854062a6bd814971c` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `bf5e227bc74abbbfbaef5f147e90ef870beea6c4b3bab0f0efcd69fb2614323c` | 2026-09-07 19:09:59 |
| 16 | resnet34 | ccsn90 | 42 | `ccsn_model_resnet34.pth` | 85,818,348 | `65729d073c11ff0dcc04e1b1c693542ffd6095e7833ce58fe29d624a5e8ff8b2` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `e5dff15494a5571224c36c713f77ad1c65f9d798dcceb5b00c10db741dacda1b` | 2026-09-07 19:15:10 |
| 17 | resnet34 | ccsn90 | 43 | `ccsn_model_resnet34.pth` | 85,818,348 | `88b1fb31b592cc3fb2582d2f2da21676e9cc6a5d1f2bf34dd9cd3a0e24fecede` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `07c9a330063ab4e8b8a77936203f1a93813f51150b50d71fee9160b15332f0f3` | 2026-09-07 19:21:00 |
| 18 | resnet34 | ccsn90 | 44 | `ccsn_model_resnet34.pth` | 85,818,348 | `37d951ab836b2970a1508edc586796a1689ac6f2a296a2c0e2c452e2fe0522f5` | `ccsn_model_resnet34_best_f1.pth` | 85,820,244 | `757527e4835460642c09613f8cce375e2c6bb749acf7e2f92986043600329cb1` | 2026-09-07 19:26:39 |
| 19 | resnet34 | gcd15 | 42 | `gcd_model_resnet34.pth` | 85,818,119 | `d2442949b8e86436038a020858db626e65ad517d8dd39747f12c6da80425887c` | `gcd_model_resnet34_best_f1.pth` | 85,820,015 | `061c2beff83cf3c05d84c7977bf0d9195aadc5c51dbe419e1c479357475d8b13` | 2026-09-07 19:57:28 |
| 20 | resnet34 | gcd15 | 43 | `gcd_model_resnet34.pth` | 85,818,119 | `df1407f94c905648ca3fc7e4eb2bbd5f4f56377aadcca3a10c143bc9a1734619` | `gcd_model_resnet34_best_f1.pth` | 85,820,015 | `256d9eaf9a4c4751f491a3d8861e34aedd4cdf4f3dad3217db2cb0a197e97da1` | 2026-09-07 20:10:45 |
| 21 | resnet34 | gcd15 | 44 | `gcd_model_resnet34.pth` | 85,818,119 | `818887796ea3f65ede3785a09d88c60321c6dd56093922f32b45f4378886dced` | `gcd_model_resnet34_best_f1.pth` | 85,820,015 | `4911bbd20b3714b186d69d4512608ccef74963db692723f7a6d3ccd72a27af31` | 2026-09-07 20:21:04 |
| 22 | resnet34 | joint15 | 42 | `harmonized_joint_resnet34.pth` | 85,819,786 | `516cef796269bc20e1b06b93e2b092f59fd967036563df2e75cea52b4939464a` | `harmonized_joint_resnet34_best_f1.pth` | 85,821,618 | `76b4b232c99f59feb7205255a066937d03ed2e737a9b4bf1b497cfa95183a003` | 2026-09-07 20:29:05 |
| 23 | resnet34 | joint15 | 43 | `harmonized_joint_resnet34.pth` | 85,819,786 | `fd4e1d7d82bea9cbcea71abd8ee1a991d7789b90c22c7e1371f7c97e01036f21` | `harmonized_joint_resnet34_best_f1.pth` | 85,821,618 | `16f605cd32e088311b9bc7c10d0fed89c4136c4c55117b1b83c966ccd8c52e0b` | 2026-09-07 20:37:20 |
| 24 | resnet34 | joint15 | 44 | `harmonized_joint_resnet34.pth` | 85,819,786 | `176983934a021879855b6973c5e963a946368599ff44c5835ac123267b7bdd19` | `harmonized_joint_resnet34_best_f1.pth` | 85,821,618 | `a96c355c77c9872a21c2e783551cefa9d6cbf2b36d2ff4d0af273ff3499120f1` | 2026-09-07 20:45:31 |
| 25 | resnet50 | ccsn15 | 42 | `ccsn_model_resnet50.pth` | 96,463,336 | `d11f287dd8660a090dd4792d1f036ca3600b9861e9466087c92bfe0c7499f9bf` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `0dee01585dd5af486e6c71e52b9aac75415c1ec1804ed880957d1c8486eb50b9` | 2026-09-07 20:46:49 |
| 26 | resnet50 | ccsn15 | 43 | `ccsn_model_resnet50.pth` | 96,463,336 | `68a29011c3262ec6c5b43bea1053f5246ad74b2017bf41f0dc459635656c876c` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `cc1712d1b151712512000cff84a877e95012381ca2195f6f89fd237b44f4e8d7` | 2026-09-07 20:47:57 |
| 27 | resnet50 | ccsn15 | 44 | `ccsn_model_resnet50.pth` | 96,463,336 | `8c2a7c74b5a38c5f7c73f645c02fadb623008ae927602eba1c0dca231aefc4e7` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `7bbcc54736dcba848ab0614137ef9f38387ca16d2f8c5705da424e35f5c1017b` | 2026-09-07 20:49:07 |
| 28 | resnet50 | ccsn90 | 42 | `ccsn_model_resnet50.pth` | 96,463,336 | `a68d36d0ac70282b28fdb8561dc9b32abfcadf8b775f06f37f81c761b1180c2c` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `b2907f4cc26ea1fcc0fd044e8817284b316651a137fad0ebea9d5bdafbc6cb1c` | 2026-09-07 20:54:05 |
| 29 | resnet50 | ccsn90 | 43 | `ccsn_model_resnet50.pth` | 96,463,336 | `0c751dfa1a263002317f22dd621adbf22b62a5bf63d7f7171e6d74d16ef7cc3a` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `7a7301c0bad4190b833a3070506a00008965775cdbbe42b7a5411b74df8ee47e` | 2026-09-07 21:00:20 |
| 30 | resnet50 | ccsn90 | 44 | `ccsn_model_resnet50.pth` | 96,463,336 | `93b64377ea9f75ae9bf656d425bc53331e0e339cf3dc7832135c480f438a0089` | `ccsn_model_resnet50_best_f1.pth` | 96,466,112 | `57c4be7356281be1e5ed0ea0da7adec5286dc80e03fc86af870636ec4baa4aa9` | 2026-09-07 21:11:26 |
| 31 | resnet50 | gcd15 | 42 | `gcd_model_resnet50.pth` | 96,463,005 | `671ab8252d55c574d02de0102a8f93733fdfc6117f2cc7a33e8f44e1eb807250` | `gcd_model_resnet50_best_f1.pth` | 96,465,781 | `f0dfc096f8b7d3500539e5fd8532a31c062500d7cb3cf551428fc1cbce9225a4` | 2026-09-07 21:20:23 |
| 32 | resnet50 | gcd15 | 43 | `gcd_model_resnet50.pth` | 96,463,005 | `83057c98f4a71ae3bf77f5d49bc6dc72d0c893b040ac41a659865671a4e5641a` | `gcd_model_resnet50_best_f1.pth` | 96,465,781 | `7a7734b016211362975157a8a58a06a49289c0731f26cf6f94859c762d70ce2c` | 2026-09-07 21:28:54 |
| 33 | resnet50 | gcd15 | 44 | `gcd_model_resnet50.pth` | 96,463,005 | `8f3fb8c317f1746cf7d85c23b0ba3a8f8136578de1bfa6bac25e95483282835e` | `gcd_model_resnet50_best_f1.pth` | 96,465,781 | `e06c0dedf52823715845d932a253a28e1f876093145779c1d8e1e1984edbee58` | 2026-09-07 21:37:24 |
| 34 | resnet50 | joint15 | 42 | `harmonized_joint_resnet50.pth` | 96,465,450 | `2f6420b4184e9ac8a05a7bf27896bc67ca05d7a3709d66d6b01d3822689559ae` | `harmonized_joint_resnet50_best_f1.pth` | 96,468,098 | `71ac42af96da8f80eebe13ea6b48f9073a00ea9346a1b4eefb467addc1616758` | 2026-09-07 21:47:14 |
| 35 | resnet50 | joint15 | 43 | `harmonized_joint_resnet50.pth` | 96,465,450 | `ce9d11ea6c4a721dee701fc1291a1cb5f927c1476d04d00d60fbebf381e4dcdf` | `harmonized_joint_resnet50_best_f1.pth` | 96,468,098 | `1407dfef7b02eceaf69e102416dd3e7b42299f7a551969941aee198afc982621` | 2026-09-07 21:57:14 |
| 36 | resnet50 | joint15 | 44 | `harmonized_joint_resnet50.pth` | 96,465,450 | `a97fd73d4b90237a0972d2034fc7dcfe1fd4c8529fd81fed1d191e6bd59ed911` | `harmonized_joint_resnet50_best_f1.pth` | 96,468,098 | `e03518c87d9d9b9e8ecb140cc8ace60c58c73f804407d58b3cce46a71791003e` | 2026-09-07 22:07:15 |

### 1.3 Checkpoint Metadata Sidecars Physical Evidence (`.meta.json`) (36 Runs)

| # | Arch | Arm | Seed | Min-Loss Meta File | Size | Min-Loss Meta SHA-256 | Max-F1 Meta File | Size | Max-F1 Meta SHA-256 |
| :-: | :--- | :--- | :-: | :--- | :-: | :--- | :--- | :-: | :--- |
| 1 | resnet18 | ccsn15 | 42 | `ccsn_model_resnet18.meta.json` | 974 B | `5473a31c8e2fc799b7afa6388128a107eb6673ff501cc5b987e1bd927e303620` | `ccsn_model_resnet18_best_f1.meta.json` | 986 B | `ced98ed8b179bf29b482920f1444f43f3b125f0d7e96e88e47e6c28380a239ba` |
| 2 | resnet18 | ccsn15 | 43 | `ccsn_model_resnet18.meta.json` | 975 B | `e282d971178521ef1362566667546d1f773737b9e9c0d057ddffdd194964aa03` | `ccsn_model_resnet18_best_f1.meta.json` | 988 B | `39f5ef49892b65f25d88fb4f9bbb2ee6f68557dd2072fb90e25793c959c1c4d7` |
| 3 | resnet18 | ccsn15 | 44 | `ccsn_model_resnet18.meta.json` | 974 B | `a60540a95a69a56809f7df9e1acd91f19f4e35a1bd1c153b88ed72eed6a6f71c` | `ccsn_model_resnet18_best_f1.meta.json` | 988 B | `ed12a153da3a3daffca504460f5f72500e12962da2dc2c0aba9b420d93700fde` |
| 4 | resnet18 | ccsn90 | 42 | `ccsn_model_resnet18.meta.json` | 976 B | `47a2f1529b620850b1a8025d13938d5509f92cb062ce3f7a235921b0d302710c` | `ccsn_model_resnet18_best_f1.meta.json` | 988 B | `ecb2b230b663a6a51f341b29c723a1b09730db786461386d99e0864aca9e2b22` |
| 5 | resnet18 | ccsn90 | 43 | `ccsn_model_resnet18.meta.json` | 975 B | `7e2cdb5c1449552868bc42a8656f233088bbb74a4a434bf739a3b404f2197500` | `ccsn_model_resnet18_best_f1.meta.json` | 987 B | `2f4d525ca78f035e4c5df81f43716d38ccad5f233300692a1e8461098e7143a6` |
| 6 | resnet18 | ccsn90 | 44 | `ccsn_model_resnet18.meta.json` | 975 B | `d69f6573aa6a19fab80c2489c42a37cd3528f2ceff084ec7848280659dd5f6ac` | `ccsn_model_resnet18_best_f1.meta.json` | 988 B | `830b1cc175d6dc1d962664fb1e487f3ca54b9cc64cd55cb3c405a15b29f9474f` |
| 7 | resnet18 | gcd15 | 42 | `gcd_model_resnet18.meta.json` | 974 B | `0d407e73c0a5af411fc2bb9a6610a5c5b00c6a7ae0e4e93e0733b2bf6e26f5b9` | `gcd_model_resnet18_best_f1.meta.json` | 986 B | `78f86c76190dc8128793a0779eeaf14db52b25d7f877204ac3849c8b9c8fcdb2` |
| 8 | resnet18 | gcd15 | 43 | `gcd_model_resnet18.meta.json` | 974 B | `5354c96ddc305529220218cc8d19d9ff03a6cb06bcc52949b479261aaf799f1b` | `gcd_model_resnet18_best_f1.meta.json` | 986 B | `1b547201b928f18ea50b13de1ec1aacf54e4372cbdc29032dc72c1b50d28fd15` |
| 9 | resnet18 | gcd15 | 44 | `gcd_model_resnet18.meta.json` | 974 B | `ca10e0a963d75fe324e81f7202fbb4af1341bfaeec4b70432d6b902d8cb6e64c` | `gcd_model_resnet18_best_f1.meta.json` | 984 B | `9e8750c22002a6c51fc6f8c49ce4253bbd2a76bc842b61a69d1a8d012f8e136e` |
| 10 | resnet18 | joint15 | 42 | `harmonized_joint_resnet18.meta.json` | 979 B | `e161e91a98a0da7a733388949bde9ebfeca6ff25a20bee7783705f2a4be3269b` | `harmonized_joint_resnet18_best_f1.meta.json` | 991 B | `b2e6aec63483aea5578b93df14cfeb2a62af81274475132d2123846a0111520e` |
| 11 | resnet18 | joint15 | 43 | `harmonized_joint_resnet18.meta.json` | 975 B | `8c3b7fb0365b92076a6057ee864e8e187adafebc9aa971ff82da3ffff627f582` | `harmonized_joint_resnet18_best_f1.meta.json` | 991 B | `280b0377a2340569c2410c438853d43cbdf8d400daf79b84ea424eb68e1c0d73` |
| 12 | resnet18 | joint15 | 44 | `harmonized_joint_resnet18.meta.json` | 977 B | `82853c1d0d46e1f22b30d944e12f71bed83fd3202720ce7fe55eb9e072aea5ee` | `harmonized_joint_resnet18_best_f1.meta.json` | 991 B | `a64e1b63f1d75c42a21990231776b59b3571c345d64f0a9a6f9ce73c9ee1a465` |
| 13 | resnet34 | ccsn15 | 42 | `ccsn_model_resnet34.meta.json` | 973 B | `0190789bfa23c3f8daacf34447eced4153726c2355dc0a9de4cd7f0205a4ab6f` | `ccsn_model_resnet34_best_f1.meta.json` | 986 B | `9f4f2fd3e23a09a889214784514351b23525cc4ff0cfb0771786d5a9dfd8ae04` |
| 14 | resnet34 | ccsn15 | 43 | `ccsn_model_resnet34.meta.json` | 973 B | `416afe5e031c7202ded149db8e6e3e402ce0ad7b564989a1958ba6de5cb04d3e` | `ccsn_model_resnet34_best_f1.meta.json` | 988 B | `307d61f382b4cc68d3b459e8fdbeca4dc80bffe1202b4f387ae39daa6591a81c` |
| 15 | resnet34 | ccsn15 | 44 | `ccsn_model_resnet34.meta.json` | 974 B | `b2c27abda4ca8c53a2285319431bbbb96f29b45e9ce9981de50404615f6081f9` | `ccsn_model_resnet34_best_f1.meta.json` | 986 B | `216453b56344f63145f83a722236ea75a3992dd1bc14b7f8017731679618abbd` |
| 16 | resnet34 | ccsn90 | 42 | `ccsn_model_resnet34.meta.json` | 974 B | `58196a9af168616827508f05e31f9110014a6850b0b47c4c72a720f2ab637afd` | `ccsn_model_resnet34_best_f1.meta.json` | 987 B | `99d5f1030231f62072b8e09cdafc1531a506b94951e20545cf048b8abe3f4dc6` |
| 17 | resnet34 | ccsn90 | 43 | `ccsn_model_resnet34.meta.json` | 974 B | `7ec25bad8c68ab13a44dec0ec6fe391c1398e598fc25cebbf610697d00290211` | `ccsn_model_resnet34_best_f1.meta.json` | 986 B | `e09e29752610ed8b325f13c72df84b0762e0bfc41f968627ecd731ff2e695bf6` |
| 18 | resnet34 | ccsn90 | 44 | `ccsn_model_resnet34.meta.json` | 974 B | `738971cdc27b317524b26d952c9d719658f7ef4a6368af9f952b8c01b9b90547` | `ccsn_model_resnet34_best_f1.meta.json` | 987 B | `01b19ee89b5ab9590518b2aaed593c3a01f8397ce2494a99a0cd9b8aaa8ceec2` |
| 19 | resnet34 | gcd15 | 42 | `gcd_model_resnet34.meta.json` | 974 B | `cf151fe62d8eb1e91aeda442ee26bb683916dd6cdf41890e42ccc39ded8ce03e` | `gcd_model_resnet34_best_f1.meta.json` | 985 B | `28ff14d36a1c1b8269596ea2a698b0f1390bb1946b7b059be4a20a26914bf9f0` |
| 20 | resnet34 | gcd15 | 43 | `gcd_model_resnet34.meta.json` | 972 B | `109bc4701b17313750cd603e2ebaf6408cfb03b16406e998b7ed8c65cbf5d9f3` | `gcd_model_resnet34_best_f1.meta.json` | 984 B | `259f1d64df951cfc2fc8083f11d16552e1e6aa236782fc67d237b93538001a42` |
| 21 | resnet34 | gcd15 | 44 | `gcd_model_resnet34.meta.json` | 974 B | `74f15055e12f8ad8cffce7f29334efd7e643a2d022cdbd96a2645352fdfbb61e` | `gcd_model_resnet34_best_f1.meta.json` | 986 B | `26696b55bb5e1dbe0cacecbd4425f9f5dfb9e264d294f21192ab05ff7507cde5` |
| 22 | resnet34 | joint15 | 42 | `harmonized_joint_resnet34.meta.json` | 979 B | `ab9f19efe7b3c121981a0e3040c1f408b648b237719f1a1df97521ea45500581` | `harmonized_joint_resnet34_best_f1.meta.json` | 991 B | `84d15836a250dc8cedcd73caea16f2fb3570b903d3850e4cc72ba0bc6b9ca25f` |
| 23 | resnet34 | joint15 | 43 | `harmonized_joint_resnet34.meta.json` | 977 B | `a1e792f76905614ce95cb71a6df0ed84909a9c38df39a94984e48c87a80f2092` | `harmonized_joint_resnet34_best_f1.meta.json` | 990 B | `58501fdc66e6017a2fbd8263160f391d003111e9f556fb09699967a53efe8fe9` |
| 24 | resnet34 | joint15 | 44 | `harmonized_joint_resnet34.meta.json` | 978 B | `469a15a5b9801cf23c84679a6aa830ce1eea917cde0925a4b9d564061a196f0c` | `harmonized_joint_resnet34_best_f1.meta.json` | 991 B | `19b639b454be68cfc28e35aea2211029c2f93215549a7e107a86c9fa0fb9b958` |
| 25 | resnet50 | ccsn15 | 42 | `ccsn_model_resnet50.meta.json` | 973 B | `7d24e2a7e5fa979d5ae35a9bc7b169ea2072fe4982c6daf645648a5fbb808ccf` | `ccsn_model_resnet50_best_f1.meta.json` | 986 B | `3e6be39ad2875670999ecbe9ccaf7d980079ae129e5446146f2b6c2b9aed9f27` |
| 26 | resnet50 | ccsn15 | 43 | `ccsn_model_resnet50.meta.json` | 974 B | `f451be58b5232b92d87a164b58a249bc9ed4bb42808d20f55ba4c4c02cc8d305` | `ccsn_model_resnet50_best_f1.meta.json` | 986 B | `a97616c9dc10f46c0c7fa04985df9584ebef1544238923894869c91a6e3d305e` |
| 27 | resnet50 | ccsn15 | 44 | `ccsn_model_resnet50.meta.json` | 974 B | `f4fb020f7f84054cda8e4312ad11f6462df5ecebf059c15e0eefd87f665d7b39` | `ccsn_model_resnet50_best_f1.meta.json` | 985 B | `64276b3ec3a9002458f94de726c71093c44ad1f27ef0a419e26c10769ccda212` |
| 28 | resnet50 | ccsn90 | 42 | `ccsn_model_resnet50.meta.json` | 974 B | `7199b323ee3b0a32896b41edd9cea75b75897412bdeadbf2fb2ddc5383069097` | `ccsn_model_resnet50_best_f1.meta.json` | 987 B | `231683cee72dda7230b61679130fadb1bd0b1149de3be9436d21d98867f22157` |
| 29 | resnet50 | ccsn90 | 43 | `ccsn_model_resnet50.meta.json` | 974 B | `ee262b533eec4f44d1c0e6d95f529043d743fba9fe5655f3cba4f1159c56ecee` | `ccsn_model_resnet50_best_f1.meta.json` | 986 B | `e178838bb19d1ed2f6dc3c9d61b0b5c353fdfa4d86b31869464c72ab6e086c20` |
| 30 | resnet50 | ccsn90 | 44 | `ccsn_model_resnet50.meta.json` | 974 B | `fff8e04c7dd8f6103f417fd48800282e1211fef3d2f5bb3e9caf37d16e0f0039` | `ccsn_model_resnet50_best_f1.meta.json` | 988 B | `41185a619b96e83f82c1850af1b2eef2ceec38aaf7e8fe5c9038890295baa433` |
| 31 | resnet50 | gcd15 | 42 | `gcd_model_resnet50.meta.json` | 974 B | `85005608d481f72d11888e8b1b7f55531b2a0828f0ea3009d34f867498510773` | `gcd_model_resnet50_best_f1.meta.json` | 986 B | `65356a82bbeb11c66dfce0dff3395c1f3b7d5bddd5f3368eae5c878c494b3223` |
| 32 | resnet50 | gcd15 | 43 | `gcd_model_resnet50.meta.json` | 972 B | `69491f476b9da16896301a076d402489602bd751d8b2eedf1cee4ca6d7ee63d0` | `gcd_model_resnet50_best_f1.meta.json` | 984 B | `fe2461b043c9f5ddadd6934c7cae6f0671c517d54478d44e43b70bd7f17a3325` |
| 33 | resnet50 | gcd15 | 44 | `gcd_model_resnet50.meta.json` | 972 B | `e1f56d514531c13f6f286482daae54f82753bf0ed11e38d2dd98462a30e13a8f` | `gcd_model_resnet50_best_f1.meta.json` | 986 B | `e8f91c654a59256f21a359371b5d63131120bc68bdb9f091db27b84b92a4045f` |
| 34 | resnet50 | joint15 | 42 | `harmonized_joint_resnet50.meta.json` | 977 B | `34d97fb91d50a96bfe5cfefa53cf021d821ebcbdab589372a7656b7fab47256d` | `harmonized_joint_resnet50_best_f1.meta.json` | 991 B | `d04f580e2bd73d0e3a7f5c5d20077b2441cb868a58b981f8cfa738e34f9aa393` |
| 35 | resnet50 | joint15 | 43 | `harmonized_joint_resnet50.meta.json` | 977 B | `e5ae910f68e184abf2850942951b749f188e0f1852ca2fee585aaee42f75b209` | `harmonized_joint_resnet50_best_f1.meta.json` | 990 B | `3a163e8451cfad51d74541313dacc6f2a645b26d1be0877f05cfddffed1bbc9a` |
| 36 | resnet50 | joint15 | 44 | `harmonized_joint_resnet50.meta.json` | 977 B | `f6d83918e9421ad4754aea04c92cbf85da318347bc5cd252c02a10ada7381697` | `harmonized_joint_resnet50_best_f1.meta.json` | 991 B | `c91575991156ece11edde094bff8f13733b74060031da68780ef8e292fa16bd0` |

### 1.4 Provenance, Seed Integrity, & Execution Commands (36 Runs)

| # | Arch | Arm | Seed | Intended Seed | Recorded Seed | Match? | Config Epochs | Recorded Epochs | Match? | Mode | Device Recorded | Git Commit | Checkpoint Mtime | Summary Mtime | Command Executed |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | resnet18 | ccsn15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 17:49:45 | 2026-09-07 17:50:08 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 42 --config config/training/tuned_resnet18_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn15_seed42` |
| 2 | resnet18 | ccsn15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 17:51:47 | 2026-09-07 17:52:10 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 43 --config config/training/tuned_resnet18_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn15_seed43` |
| 3 | resnet18 | ccsn15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 17:53:10 | 2026-09-07 17:53:33 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 44 --config config/training/tuned_resnet18_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn15_seed44` |
| 4 | resnet18 | ccsn90 | 42 | 42 | 42 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 17:57:29 | 2026-09-07 17:57:52 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 42 --config config/training/tuned_resnet18_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn90_seed42` |
| 5 | resnet18 | ccsn90 | 43 | 43 | 43 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:01:20 | 2026-09-07 18:01:43 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 43 --config config/training/tuned_resnet18_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn90_seed43` |
| 6 | resnet18 | ccsn90 | 44 | 44 | 44 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:05:11 | 2026-09-07 18:05:34 | `python src/run_harmonized.py --experiment ccsn --model resnet18 --seed 44 --config config/training/tuned_resnet18_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet18/ccsn90_seed44` |
| 7 | resnet18 | gcd15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:09:14 | 2026-09-07 18:09:37 | `python src/run_harmonized.py --experiment gcd --model resnet18 --seed 42 --config config/training/tuned_resnet18_gcd.toml --output-dir artifacts/harmonized_thorough/resnet18/gcd15_seed42` |
| 8 | resnet18 | gcd15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:13:18 | 2026-09-07 18:13:41 | `python src/run_harmonized.py --experiment gcd --model resnet18 --seed 43 --config config/training/tuned_resnet18_gcd.toml --output-dir artifacts/harmonized_thorough/resnet18/gcd15_seed43` |
| 9 | resnet18 | gcd15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:17:18 | 2026-09-07 18:17:41 | `python src/run_harmonized.py --experiment gcd --model resnet18 --seed 44 --config config/training/tuned_resnet18_gcd.toml --output-dir artifacts/harmonized_thorough/resnet18/gcd15_seed44` |
| 10 | resnet18 | joint15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:21:58 | 2026-09-07 18:22:35 | `python src/run_harmonized.py --experiment joint --model resnet18 --seed 42 --config config/training/tuned_resnet18.toml --output-dir artifacts/harmonized_thorough/resnet18/joint_seed42` |
| 11 | resnet18 | joint15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:26:55 | 2026-09-07 18:27:31 | `python src/run_harmonized.py --experiment joint --model resnet18 --seed 43 --config config/training/tuned_resnet18.toml --output-dir artifacts/harmonized_thorough/resnet18/joint_seed43` |
| 12 | resnet18 | joint15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 18:31:45 | 2026-09-07 18:32:22 | `python src/run_harmonized.py --experiment joint --model resnet18 --seed 44 --config config/training/tuned_resnet18.toml --output-dir artifacts/harmonized_thorough/resnet18/joint_seed44` |
| 13 | resnet34 | ccsn15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:06:28 | 2026-09-07 19:07:00 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 42 --config config/training/tuned_resnet34_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn15_seed42` |
| 14 | resnet34 | ccsn15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:08:13 | 2026-09-07 19:08:45 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 43 --config config/training/tuned_resnet34_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn15_seed43` |
| 15 | resnet34 | ccsn15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:09:59 | 2026-09-07 19:10:31 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 44 --config config/training/tuned_resnet34_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn15_seed44` |
| 16 | resnet34 | ccsn90 | 42 | 42 | 42 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:15:10 | 2026-09-07 19:15:43 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 42 --config config/training/tuned_resnet34_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn90_seed42` |
| 17 | resnet34 | ccsn90 | 43 | 43 | 43 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:21:00 | 2026-09-07 19:21:33 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 43 --config config/training/tuned_resnet34_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn90_seed43` |
| 18 | resnet34 | ccsn90 | 44 | 44 | 44 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:26:39 | 2026-09-07 19:27:11 | `python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 44 --config config/training/tuned_resnet34_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet34/ccsn90_seed44` |
| 19 | resnet34 | gcd15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 19:57:28 | 2026-09-07 19:57:53 | `python src/run_harmonized.py --experiment gcd --model resnet34 --seed 42 --config config/training/tuned_resnet34_gcd.toml --output-dir artifacts/harmonized_thorough/resnet34/gcd15_seed42` |
| 20 | resnet34 | gcd15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:10:45 | 2026-09-07 20:11:09 | `python src/run_harmonized.py --experiment gcd --model resnet34 --seed 43 --config config/training/tuned_resnet34_gcd.toml --output-dir artifacts/harmonized_thorough/resnet34/gcd15_seed43` |
| 21 | resnet34 | gcd15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:21:04 | 2026-09-07 20:21:16 | `python src/run_harmonized.py --experiment gcd --model resnet34 --seed 44 --config config/training/tuned_resnet34_gcd.toml --output-dir artifacts/harmonized_thorough/resnet34/gcd15_seed44` |
| 22 | resnet34 | joint15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:29:05 | 2026-09-07 20:29:28 | `python src/run_harmonized.py --experiment joint --model resnet34 --seed 42 --config config/training/tuned_resnet34.toml --output-dir artifacts/harmonized_thorough/resnet34/joint_seed42` |
| 23 | resnet34 | joint15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:37:20 | 2026-09-07 20:37:43 | `python src/run_harmonized.py --experiment joint --model resnet34 --seed 43 --config config/training/tuned_resnet34.toml --output-dir artifacts/harmonized_thorough/resnet34/joint_seed43` |
| 24 | resnet34 | joint15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:45:31 | 2026-09-07 20:45:53 | `python src/run_harmonized.py --experiment joint --model resnet34 --seed 44 --config config/training/tuned_resnet34.toml --output-dir artifacts/harmonized_thorough/resnet34/joint_seed44` |
| 25 | resnet50 | ccsn15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:46:49 | 2026-09-07 20:47:02 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 42 --config config/training/tuned_resnet50_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn15_seed42` |
| 26 | resnet50 | ccsn15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:47:57 | 2026-09-07 20:48:11 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 43 --config config/training/tuned_resnet50_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn15_seed43` |
| 27 | resnet50 | ccsn15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:49:07 | 2026-09-07 20:49:20 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 44 --config config/training/tuned_resnet50_ccsn_15ep.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn15_seed44` |
| 28 | resnet50 | ccsn90 | 42 | 42 | 42 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 20:54:05 | 2026-09-07 20:54:19 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 42 --config config/training/tuned_resnet50_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn90_seed42` |
| 29 | resnet50 | ccsn90 | 43 | 43 | 43 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:00:20 | 2026-09-07 21:00:45 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 43 --config config/training/tuned_resnet50_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn90_seed43` |
| 30 | resnet50 | ccsn90 | 44 | 44 | 44 | **MATCH** | 90 | 90 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:11:26 | 2026-09-07 21:11:40 | `python src/run_harmonized.py --experiment ccsn --model resnet50 --seed 44 --config config/training/tuned_resnet50_ccsn.toml --output-dir artifacts/harmonized_thorough/resnet50/ccsn90_seed44` |
| 31 | resnet50 | gcd15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:20:23 | 2026-09-07 21:20:37 | `python src/run_harmonized.py --experiment gcd --model resnet50 --seed 42 --config config/training/tuned_resnet50_gcd.toml --output-dir artifacts/harmonized_thorough/resnet50/gcd15_seed42` |
| 32 | resnet50 | gcd15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:28:54 | 2026-09-07 21:29:08 | `python src/run_harmonized.py --experiment gcd --model resnet50 --seed 43 --config config/training/tuned_resnet50_gcd.toml --output-dir artifacts/harmonized_thorough/resnet50/gcd15_seed43` |
| 33 | resnet50 | gcd15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:37:24 | 2026-09-07 21:37:38 | `python src/run_harmonized.py --experiment gcd --model resnet50 --seed 44 --config config/training/tuned_resnet50_gcd.toml --output-dir artifacts/harmonized_thorough/resnet50/gcd15_seed44` |
| 34 | resnet50 | joint15 | 42 | 42 | 42 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:47:14 | 2026-09-07 21:47:38 | `python src/run_harmonized.py --experiment joint --model resnet50 --seed 42 --config config/training/tuned_resnet50.toml --output-dir artifacts/harmonized_thorough/resnet50/joint_seed42` |
| 35 | resnet50 | joint15 | 43 | 43 | 43 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 21:57:14 | 2026-09-07 21:57:38 | `python src/run_harmonized.py --experiment joint --model resnet50 --seed 43 --config config/training/tuned_resnet50.toml --output-dir artifacts/harmonized_thorough/resnet50/joint_seed43` |
| 36 | resnet50 | joint15 | 44 | 44 | 44 | **MATCH** | 15 | 15 | **MATCH** | Newly Trained | None (RTX 4070 in env) | None | 2026-09-07 22:07:15 | 2026-09-07 22:07:39 | `python src/run_harmonized.py --experiment joint --model resnet50 --seed 44 --config config/training/tuned_resnet50.toml --output-dir artifacts/harmonized_thorough/resnet50/joint_seed44` |

---

## 2. Configuration File Inventory

Inventory of all TOML configuration files located in `d:\cloud-classification-cnn-mobile\config\training\`.

| Configuration File | Absolute Path | Exists | Size (Bytes) | Epochs | SHA-256 Checksum | Modification Time (mtime) | Usage / Benchmark Role |
| :--- | :--- | :-: | :-: | :-: | :--- | :--- | :--- |
| `baseline.toml` | `d:\cloud-classification-cnn-mobile\config\training\baseline.toml` | **Yes** | 411 | 10 | `917390e6045d2c5285e6db228ce52890afec9dbb4d896d895db99ebb0921a7f6` | 2026-09-06 03:39:52 | Default baseline configuration (not used in harmonized_thorough) |
| `smoke.toml` | `d:\cloud-classification-cnn-mobile\config\training\smoke.toml` | **Yes** | 258 | 1 | `941887b938ab07a570dd887684a7bf71f939d9f87cc23b9b32e833f502b1945e` | 2026-08-30 22:15:38 | Fast smoke test configuration (not used in harmonized_thorough) |
| `tuned_resnet18.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18.toml` | **Yes** | 781 | 15 | `f8c8021475f5691004aa6fab60078e4f219bc9b360245578f5669889ae4e7e64` | 2026-09-06 03:24:24 | ResNet-18 Joint-15ep production runs (passed via --config) |
| `tuned_resnet18_ccsn.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18_ccsn.toml` | **Yes** | 1,200 | 90 | `671005513539d544e206d8d4f3428c5b8e4a4a02a6aeeb1f483dcd8a6f67f7e0` | 2026-09-07 14:59:01 | ResNet-18 CCSN-90ep budget-matched production runs |
| `tuned_resnet18_ccsn_15ep.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18_ccsn_15ep.toml` | **Yes** | 1,156 | 15 | `5767f5e948fea12f32261f1459eb3dd753198dce4eef5af86b6c0e1b8423acd2` | 2026-09-07 16:19:08 | ResNet-18 CCSN-15ep control production runs |
| `tuned_resnet18_ccsn_budget_matched.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18_ccsn_budget_matched.toml` | **Yes** | 2,889 | 15 | `54291b51632e58f91a08cb2eff553247711a8973521b0d99031f6af0f43d9876` | 2026-09-07 14:48:00 | Legacy preliminary sweep configuration (not used in harmonized_thorough) |
| `tuned_resnet18_gcd.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18_gcd.toml` | **Yes** | 917 | 15 | `b333f5476eb72a1c1b38bd9d79474beb7b823f63d6650ed71d5236436868e1e3` | 2026-09-07 15:20:40 | ResNet-18 GCD-15ep production runs |
| `tuned_resnet18_joint.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet18_joint.toml` | **Yes** | 800 | 15 | `5b5acc797d6b12d899638e9c4bc4a5f8617ca52eb9a7f7e5b41a6b8b62a852b0` | 2026-09-07 14:59:26 | ResNet-18 Joint-15ep reference (identical parameters to tuned_resnet18.toml) |
| `tuned_resnet34.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet34.toml` | **Yes** | 781 | 15 | `3e9364e85e6588413ba3986c648483efcc0c2874a6a3e61dee4350aa6aa44287` | 2026-09-06 03:24:24 | ResNet-34 Joint-15ep production runs (passed via --config) |
| `tuned_resnet34_ccsn.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet34_ccsn.toml` | **Yes** | 1,208 | 90 | `46f71c34c0910f463d69cd6fa75de76ec3edffa07fcf2395c25792c4d2f8eff8` | 2026-09-07 18:40:37 | ResNet-34 CCSN-90ep budget-matched production runs |
| `tuned_resnet34_ccsn_15ep.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet34_ccsn_15ep.toml` | **Yes** | 1,164 | 15 | `30fa8bb5bcf0e374845a14368d5f060fc45b639daf57248ecb33ff5e4c3d59f6` | 2026-09-07 19:05:16 | ResNet-34 CCSN-15ep control production runs |
| `tuned_resnet34_gcd.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet34_gcd.toml` | **Yes** | 918 | 15 | `a744cd23ccd8c5b5de684613318d65f797806174d6970e695970e456d23c988a` | 2026-09-07 18:49:02 | ResNet-34 GCD-15ep production runs |
| `tuned_resnet50.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet50.toml` | **Yes** | 781 | 15 | `a615b2efec32cf61c2e7a43d64709d30d2c8ca73a59dae931ef9283c47a57f05` | 2026-09-06 03:24:24 | ResNet-50 Joint-15ep production runs (passed via --config) |
| `tuned_resnet50_ccsn.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet50_ccsn.toml` | **Yes** | 1,208 | 90 | `890e9d8d1996613438fdd12d98a1374866fee1766103399b3c87aab39665f85f` | 2026-09-07 18:51:34 | ResNet-50 CCSN-90ep budget-matched production runs |
| `tuned_resnet50_ccsn_15ep.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet50_ccsn_15ep.toml` | **Yes** | 1,164 | 15 | `d4a01a8416c37edcd9e5572442b5216be3235a05c9b718dcf8c3d6c04feeb8f4` | 2026-09-07 19:05:16 | ResNet-50 CCSN-15ep control production runs |
| `tuned_resnet50_gcd.toml` | `d:\cloud-classification-cnn-mobile\config\training\tuned_resnet50_gcd.toml` | **Yes** | 918 | 15 | `d6b8fa3779bc6fd0936f171883b8bae02defb3bbb218b2bab9d1e73b5cd03f4e` | 2026-09-07 19:05:13 | ResNet-50 GCD-15ep production runs |

### 2.1 Resolution of Joint Configuration Files (`resnet34` & `resnet50` vs `resnet18`)

A physical inspection of the filesystem and orchestration scripts establishes the exact resolution mechanism for joint training configurations:

1. **ResNet-18 Joint Configuration**: Both `config/training/tuned_resnet18.toml` and `config/training/tuned_resnet18_joint.toml` physically exist on disk. A byte-level inspection confirms they specify **identical hyperparameters**: AdamW optimizer, backbone learning rate $5\times 10^{-5}$, classification head learning rate $5\times 10^{-4}$, weight decay $0.01$, label smoothing $0.0$, backbone dropout $0.3$, head dropout $0.2$, cosine annealing scheduler, 15 epochs, batch size 64. The production orchestrator executed `tuned_resnet18.toml`.
2. **ResNet-34 and ResNet-50 Joint Configurations**: There are **no files named `tuned_resnet34_joint.toml` or `tuned_resnet50_joint.toml`** on disk, and none were ever created. During Phase 1 hyperparameter tuning, tuning sweeps on the combined joint pool automatically produced `config/training/tuned_resnet34.toml` and `config/training/tuned_resnet50.toml` respectively as the canonical tuned joint models. In the production orchestrator (`run_full_sweep_pipeline.py`, lines 88–89), the Joint-15ep arm maps directly to `--config config/training/tuned_{arch}.toml`. Therefore, `tuned_resnet34.toml` and `tuned_resnet50.toml` are the exact configuration files used for the ResNet-34 and ResNet-50 Joint-15ep runs.
3. **CCSN-15ep Control Configuration**: For all three architectures, the 15-epoch control arm used `config/training/tuned_{arch}_ccsn_15ep.toml`, which was derived from `tuned_{arch}_ccsn.toml` by adjusting the epoch budget from 90 to 15 epochs (360 total steps instead of 2,160 steps) while maintaining identical architectural and optimization hyperparameters.

---

## 3. Compiled Output Verification

Verification of master aggregated datasets, compiled markdown reports, compute environment metadata, and Phase 1 hyperparameter tuning trajectory files.

### 3.1 Aggregated Files & Master Deliverables

| Deliverable | Absolute Path | Exists | Byte Size | SHA-256 Checksum | Modification Time | Aggregation Details |
| :--- | :--- | :-: | :-: | :--- | :--- | :--- |
| `master_multi_seed_results.json` | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\master_multi_seed_results.json` | **Yes** | 525,629 | `e102098f2ee9ee0ed01eff1c4d1119ac2d09861006eec00f284c519a62350b9e` | 2026-09-07 22:07:40 | 36 unique benchmark runs aggregated (36 runs total across 2 criteria) |
| `master_multi_seed_report.md` | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\master_multi_seed_report.md` | **Yes** | 10,651 | `4f7d3fb05e2f7643a603b68d2106fa3b4a3a2fc630ba7303cccbb61b0ac5b633` | 2026-09-07 22:07:40 |  |
| `compute_environment.json` | `d:\cloud-classification-cnn-mobile\artifacts\harmonized_thorough\compute_environment.json` | **Yes** | 281 | `30cfcc41767f56f6291718430173a9949b0eca30f39658b9317a1b50618b4f51` | 2026-09-07 21:57:42 |  |

### 3.2 Phase 1 Hyperparameter Tuning Sweep Trajectories (`artifacts/tuning/`)

All Phase 1 hyperparameter tuning sweeps generated trajectory logs containing epoch-by-epoch loss and validation metrics across all evaluated trials under `artifacts/tuning/`:

| Tuning Trajectory JSON | Absolute Path | Exists | Byte Size | SHA-256 Checksum | Modification Time |
| :--- | :--- | :-: | :-: | :--- | :--- |
| `tuning_results_resnet18.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet18.json` | **Yes** | 12,850 | `9d816de03ffd58fcc36f044880cce0386c72ce0c28f69dc2333d7bb3a2ed66ca` | 2026-09-05 05:11:06 |
| `tuning_results_resnet18_ccsn.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet18_ccsn.json` | **Yes** | 15,141 | `24acbcc498667fb2789898b782522a2011d0ffe300231352add9e8b686634695` | 2026-09-07 14:59:01 |
| `tuning_results_resnet18_gcd.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet18_gcd.json` | **Yes** | 15,165 | `7009332a3fd6c6f216c736474ec05992b2d64b4a5c71ee73a4dfe7d3827ae66f` | 2026-09-07 15:20:39 |
| `tuning_results_resnet34.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet34.json` | **Yes** | 12,846 | `9f8668e870ffe1516e88269034464ece7e185f786fa1eb3bc99f1382618dc003` | 2026-09-05 05:36:58 |
| `tuning_results_resnet34_ccsn.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet34_ccsn.json` | **Yes** | 15,156 | `7e77deffa5d43d7326610af5379394f748b9a077d915f879d18df9c5e0d2c2d4` | 2026-09-07 18:40:36 |
| `tuning_results_resnet34_gcd.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet34_gcd.json` | **Yes** | 15,160 | `e0a4b98b2e73c4a6126b49a9853b91a4eb307b4eb80ebe2ec76075ffb639d9f0` | 2026-09-07 18:49:01 |
| `tuning_results_resnet50.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet50.json` | **Yes** | 12,840 | `fb291ef02d564f9d86258d4b8710a72681646bbde2ee6bd49ae2c398a171faff` | 2026-09-05 19:16:42 |
| `tuning_results_resnet50_ccsn.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet50_ccsn.json` | **Yes** | 15,165 | `70a015e4f05cfce7f8a9220dd9fd95e93bb834cf221899f89ca372e2222b4f42` | 2026-09-07 18:51:34 |
| `tuning_results_resnet50_gcd.json` | `d:\cloud-classification-cnn-mobile\artifacts\tuning\tuning_results_resnet50_gcd.json` | **Yes** | 15,151 | `23dcc366fab41f3e699bac0e00027b5dbd4f3814e9780d2bee7444916e741c8a` | 2026-09-07 19:05:12 |

---

## 4. Prediction and Confusion Matrix Artifacts Assessment

A physical disk inspection was conducted across all 36 run directories (`artifacts/harmonized_thorough/{arch}/{arm}_seed{seed}/`) to locate per-sample predictions, machine-readable confusion matrices, and figure plots.

### 4.1 Per-Run Artifact Availability Audit

| Architecture | Arm | Seed | Per-Sample Predictions (`y_true`, `y_pred`, `y_prob`) | Confusion Matrix Data (`.npy`, `.csv`, `.json`) | Per-Run Confusion Matrix Plot (`.png`, `.svg`) |
| :--- | :--- | :-: | :---: | :---: | :---: |
| resnet18 | ccsn15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet18 | ccsn15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet18 | ccsn15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet18 | ccsn90 | 42 | **NONE** | **NONE** | **NONE** |
| resnet18 | ccsn90 | 43 | **NONE** | **NONE** | **NONE** |
| resnet18 | ccsn90 | 44 | **NONE** | **NONE** | **NONE** |
| resnet18 | gcd15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet18 | gcd15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet18 | gcd15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet18 | joint15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet18 | joint15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet18 | joint15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn90 | 42 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn90 | 43 | **NONE** | **NONE** | **NONE** |
| resnet34 | ccsn90 | 44 | **NONE** | **NONE** | **NONE** |
| resnet34 | gcd15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet34 | gcd15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet34 | gcd15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet34 | joint15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet34 | joint15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet34 | joint15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn90 | 42 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn90 | 43 | **NONE** | **NONE** | **NONE** |
| resnet50 | ccsn90 | 44 | **NONE** | **NONE** | **NONE** |
| resnet50 | gcd15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet50 | gcd15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet50 | gcd15 | 44 | **NONE** | **NONE** | **NONE** |
| resnet50 | joint15 | 42 | **NONE** | **NONE** | **NONE** |
| resnet50 | joint15 | 43 | **NONE** | **NONE** | **NONE** |
| resnet50 | joint15 | 44 | **NONE** | **NONE** | **NONE** |

### 4.2 Explicit Missing Artifact Disclosure

> [!WARNING]
> **Missing Granular Prediction Artifacts**: Neither per-sample prediction arrays (`y_true`, `y_pred`, `y_prob`) nor machine-readable confusion matrices (`cm.npy`, `cm.csv`, `cm.json`) were serialized to disk by `run_harmonized.py`. They do not exist in any of the 36 run directories.
> 
> **Shared Figure Overwriting**: Confusion matrix figure plots (`.png`) were written to a single shared directory (`artifacts/figures/`) using unseeded filenames (e.g. `confusion_matrix_harmonized_ccsn_in_domain_(five_class,_resnet18)_resnet18.png`). Consequently, each successive seed execution overwritten the previous seed's plot.
> 
> **Reproducibility Path**: Because 100% of the trained PyTorch checkpoints (`.pth`) and their configuration sidecars (`.meta.json`) are preserved on disk with cryptographic integrity, full per-sample predictions, logits, ROC curves, and distinct per-seed confusion matrices can be generated deterministically at any time by evaluating the saved model checkpoints on the canonical test splits.

---

## 5. Recursive Directory Listing of `artifacts/harmonized_thorough/`

Below is the complete recursive file tree of `artifacts/harmonized_thorough/` detailing all directories and files with their exact byte sizes:

```text
📁 artifacts/harmonized_thorough/
  📄 compute_environment.json (281 bytes)
  📄 full_pipeline.log (7,834 bytes)
  📄 master_multi_seed_report.md (10,651 bytes)
  📄 master_multi_seed_results.json (525,629 bytes)
  📁 resnet18/
    📄 progress.json (2,727 bytes)
    📁 ccsn15_seed42/
      📄 ccsn_model_resnet18.meta.json (974 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,353 bytes)
    📁 ccsn15_seed43/
      📄 ccsn_model_resnet18.meta.json (975 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,357 bytes)
    📁 ccsn15_seed44/
      📄 ccsn_model_resnet18.meta.json (974 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,357 bytes)
    📁 ccsn90_seed42/
      📄 ccsn_model_resnet18.meta.json (976 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,356 bytes)
    📁 ccsn90_seed43/
      📄 ccsn_model_resnet18.meta.json (975 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (987 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,354 bytes)
    📁 ccsn90_seed44/
      📄 ccsn_model_resnet18.meta.json (975 bytes)
      📄 ccsn_model_resnet18.pth (45,323,308 bytes)
      📄 ccsn_model_resnet18_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet18_best_f1.pth (45,324,436 bytes)
      📄 harmonized_summary_resnet18.json (16,349 bytes)
    📁 gcd15_seed42/
      📄 gcd_model_resnet18.meta.json (974 bytes)
      📄 gcd_model_resnet18.pth (45,323,175 bytes)
      📄 gcd_model_resnet18_best_f1.meta.json (986 bytes)
      📄 gcd_model_resnet18_best_f1.pth (45,324,303 bytes)
      📄 harmonized_summary_resnet18.json (16,347 bytes)
    📁 gcd15_seed43/
      📄 gcd_model_resnet18.meta.json (974 bytes)
      📄 gcd_model_resnet18.pth (45,323,175 bytes)
      📄 gcd_model_resnet18_best_f1.meta.json (986 bytes)
      📄 gcd_model_resnet18_best_f1.pth (45,324,303 bytes)
      📄 harmonized_summary_resnet18.json (16,338 bytes)
    📁 gcd15_seed44/
      📄 gcd_model_resnet18.meta.json (974 bytes)
      📄 gcd_model_resnet18.pth (45,323,175 bytes)
      📄 gcd_model_resnet18_best_f1.meta.json (984 bytes)
      📄 gcd_model_resnet18_best_f1.pth (45,324,303 bytes)
      📄 harmonized_summary_resnet18.json (16,335 bytes)
    📁 joint_seed42/
      📄 harmonized_joint_resnet18.meta.json (979 bytes)
      📄 harmonized_joint_resnet18.pth (45,324,170 bytes)
      📄 harmonized_joint_resnet18_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet18_best_f1.pth (45,325,234 bytes)
      📄 harmonized_summary_resnet18.json (22,648 bytes)
    📁 joint_seed43/
      📄 harmonized_joint_resnet18.meta.json (975 bytes)
      📄 harmonized_joint_resnet18.pth (45,324,170 bytes)
      📄 harmonized_joint_resnet18_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet18_best_f1.pth (45,325,234 bytes)
      📄 harmonized_summary_resnet18.json (22,648 bytes)
    📁 joint_seed44/
      📄 harmonized_joint_resnet18.meta.json (977 bytes)
      📄 harmonized_joint_resnet18.pth (45,324,170 bytes)
      📄 harmonized_joint_resnet18_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet18_best_f1.pth (45,325,234 bytes)
      📄 harmonized_summary_resnet18.json (22,645 bytes)
  📁 resnet34/
    📁 ccsn15_seed42/
      📄 ccsn_model_resnet34.meta.json (973 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,347 bytes)
    📁 ccsn15_seed43/
      📄 ccsn_model_resnet34.meta.json (973 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,341 bytes)
    📁 ccsn15_seed44/
      📄 ccsn_model_resnet34.meta.json (974 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,342 bytes)
    📁 ccsn90_seed42/
      📄 ccsn_model_resnet34.meta.json (974 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (987 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,348 bytes)
    📁 ccsn90_seed43/
      📄 ccsn_model_resnet34.meta.json (974 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,348 bytes)
    📁 ccsn90_seed44/
      📄 ccsn_model_resnet34.meta.json (974 bytes)
      📄 ccsn_model_resnet34.pth (85,818,348 bytes)
      📄 ccsn_model_resnet34_best_f1.meta.json (987 bytes)
      📄 ccsn_model_resnet34_best_f1.pth (85,820,244 bytes)
      📄 harmonized_summary_resnet34.json (16,339 bytes)
    📁 gcd15_seed42/
      📄 gcd_model_resnet34.meta.json (974 bytes)
      📄 gcd_model_resnet34.pth (85,818,119 bytes)
      📄 gcd_model_resnet34_best_f1.meta.json (985 bytes)
      📄 gcd_model_resnet34_best_f1.pth (85,820,015 bytes)
      📄 harmonized_summary_resnet34.json (16,336 bytes)
    📁 gcd15_seed43/
      📄 gcd_model_resnet34.meta.json (972 bytes)
      📄 gcd_model_resnet34.pth (85,818,119 bytes)
      📄 gcd_model_resnet34_best_f1.meta.json (984 bytes)
      📄 gcd_model_resnet34_best_f1.pth (85,820,015 bytes)
      📄 harmonized_summary_resnet34.json (16,327 bytes)
    📁 gcd15_seed44/
      📄 gcd_model_resnet34.meta.json (974 bytes)
      📄 gcd_model_resnet34.pth (85,818,119 bytes)
      📄 gcd_model_resnet34_best_f1.meta.json (986 bytes)
      📄 gcd_model_resnet34_best_f1.pth (85,820,015 bytes)
      📄 harmonized_summary_resnet34.json (16,337 bytes)
    📁 joint_seed42/
      📄 harmonized_joint_resnet34.meta.json (979 bytes)
      📄 harmonized_joint_resnet34.pth (85,819,786 bytes)
      📄 harmonized_joint_resnet34_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet34_best_f1.pth (85,821,618 bytes)
      📄 harmonized_summary_resnet34.json (22,660 bytes)
    📁 joint_seed43/
      📄 harmonized_joint_resnet34.meta.json (977 bytes)
      📄 harmonized_joint_resnet34.pth (85,819,786 bytes)
      📄 harmonized_joint_resnet34_best_f1.meta.json (990 bytes)
      📄 harmonized_joint_resnet34_best_f1.pth (85,821,618 bytes)
      📄 harmonized_summary_resnet34.json (22,653 bytes)
    📁 joint_seed44/
      📄 harmonized_joint_resnet34.meta.json (978 bytes)
      📄 harmonized_joint_resnet34.pth (85,819,786 bytes)
      📄 harmonized_joint_resnet34_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet34_best_f1.pth (85,821,618 bytes)
      📄 harmonized_summary_resnet34.json (22,656 bytes)
  📁 resnet50/
    📁 ccsn15_seed42/
      📄 ccsn_model_resnet50.meta.json (973 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,340 bytes)
    📁 ccsn15_seed43/
      📄 ccsn_model_resnet50.meta.json (974 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,353 bytes)
    📁 ccsn15_seed44/
      📄 ccsn_model_resnet50.meta.json (974 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (985 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,336 bytes)
    📁 ccsn90_seed42/
      📄 ccsn_model_resnet50.meta.json (974 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (987 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,345 bytes)
    📁 ccsn90_seed43/
      📄 ccsn_model_resnet50.meta.json (974 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (986 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,356 bytes)
    📁 ccsn90_seed44/
      📄 ccsn_model_resnet50.meta.json (974 bytes)
      📄 ccsn_model_resnet50.pth (96,463,336 bytes)
      📄 ccsn_model_resnet50_best_f1.meta.json (988 bytes)
      📄 ccsn_model_resnet50_best_f1.pth (96,466,112 bytes)
      📄 harmonized_summary_resnet50.json (16,352 bytes)
    📁 gcd15_seed42/
      📄 gcd_model_resnet50.meta.json (974 bytes)
      📄 gcd_model_resnet50.pth (96,463,005 bytes)
      📄 gcd_model_resnet50_best_f1.meta.json (986 bytes)
      📄 gcd_model_resnet50_best_f1.pth (96,465,781 bytes)
      📄 harmonized_summary_resnet50.json (16,332 bytes)
    📁 gcd15_seed43/
      📄 gcd_model_resnet50.meta.json (972 bytes)
      📄 gcd_model_resnet50.pth (96,463,005 bytes)
      📄 gcd_model_resnet50_best_f1.meta.json (984 bytes)
      📄 gcd_model_resnet50_best_f1.pth (96,465,781 bytes)
      📄 harmonized_summary_resnet50.json (16,335 bytes)
    📁 gcd15_seed44/
      📄 gcd_model_resnet50.meta.json (972 bytes)
      📄 gcd_model_resnet50.pth (96,463,005 bytes)
      📄 gcd_model_resnet50_best_f1.meta.json (986 bytes)
      📄 gcd_model_resnet50_best_f1.pth (96,465,781 bytes)
      📄 harmonized_summary_resnet50.json (16,336 bytes)
    📁 joint_seed42/
      📄 harmonized_joint_resnet50.meta.json (977 bytes)
      📄 harmonized_joint_resnet50.pth (96,465,450 bytes)
      📄 harmonized_joint_resnet50_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet50_best_f1.pth (96,468,098 bytes)
      📄 harmonized_summary_resnet50.json (22,654 bytes)
    📁 joint_seed43/
      📄 harmonized_joint_resnet50.meta.json (977 bytes)
      📄 harmonized_joint_resnet50.pth (96,465,450 bytes)
      📄 harmonized_joint_resnet50_best_f1.meta.json (990 bytes)
      📄 harmonized_joint_resnet50_best_f1.pth (96,468,098 bytes)
      📄 harmonized_summary_resnet50.json (22,655 bytes)
    📁 joint_seed44/
      📄 harmonized_joint_resnet50.meta.json (977 bytes)
      📄 harmonized_joint_resnet50.pth (96,465,450 bytes)
      📄 harmonized_joint_resnet50_best_f1.meta.json (991 bytes)
      📄 harmonized_joint_resnet50_best_f1.pth (96,468,098 bytes)
      📄 harmonized_summary_resnet50.json (22,653 bytes)
```


---

## 6. Reconciliation with Previous Report

This section presents a cell-by-cell numerical comparison between the values reported in the previous master report and the actual values extracted directly from the physical `harmonized_summary_{arch}.json` files on disk.

### 6.1 Statement of Reconciliation Result

> [!IMPORTANT]
> **EXACT MATCH CONFIRMED**: Every single number previously reported (top-1 accuracy, balanced accuracy, and macro-F1 across all 36 runs, for both Min Validation Loss and Max Validation Macro-F1 criteria) **matches the physical on-disk summary JSONs down to 0.01% with ZERO discrepancies** across all 378 evaluated metric points.
> 
> Recalculated 3-seed means and sample standard deviations ($ddof=1$) match the reported tables with 100.00% precision.

### 6.2 Three-Seed Aggregate Reconciliation (Min Validation Loss Criterion - Canonical)

| Architecture | Arm / Condition | Evaluated On | Reported Top-1 Acc | Recalculated Top-1 Acc | Reported Bal Acc | Recalculated Bal Acc | Reported Macro-F1 | Recalculated Macro-F1 | Discrepancy? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :-: |
| **RESNET18** | ccsn15 | ccsn | 58.90% ± 0.96% | 58.90% ± 0.96% | 54.93% ± 0.47% | 54.93% ± 0.47% | 55.28% ± 1.02% | 55.28% ± 1.02% | **NONE (Exact)** |
| **RESNET18** | ccsn90 | ccsn | 61.18% ± 0.54% | 61.18% ± 0.54% | 57.00% ± 0.25% | 57.00% ± 0.25% | 57.51% ± 0.43% | 57.51% ± 0.43% | **NONE (Exact)** |
| **RESNET18** | gcd15 | gcd | 89.88% ± 0.20% | 89.88% ± 0.20% | 92.10% ± 0.29% | 92.10% ± 0.29% | 91.74% ± 0.13% | 91.74% ± 0.13% | **NONE (Exact)** |
| **RESNET18** | joint15 | ccsn | 60.97% ± 0.54% | 60.97% ± 0.54% | 57.73% ± 0.31% | 57.73% ± 0.31% | 58.02% ± 0.03% | 58.02% ± 0.03% | **NONE (Exact)** |
| **RESNET18** | joint15 | gcd | 87.22% ± 1.77% | 87.22% ± 1.77% | 88.35% ± 1.87% | 88.35% ± 1.87% | 88.85% ± 1.81% | 88.85% ± 1.81% | **NONE (Exact)** |
| **RESNET18** | joint15 | joint | 83.53% ± 1.59% | 83.53% ± 1.59% | — | — | — | — | **NONE (Exact)** |
| **RESNET18** | joint15 | source_balanced_average | 74.09% ± 1.14% | 74.09% ± 1.14% | — | — | — | — | **NONE (Exact)** |
| **RESNET34** | ccsn15 | ccsn | 58.05% ± 1.78% | 58.05% ± 1.78% | 55.74% ± 0.65% | 55.74% ± 0.65% | 55.35% ± 0.88% | 55.35% ± 0.88% | **NONE (Exact)** |
| **RESNET34** | ccsn90 | ccsn | 57.48% ± 1.62% | 57.48% ± 1.62% | 55.12% ± 0.70% | 55.12% ± 0.70% | 54.79% ± 0.50% | 54.79% ± 0.50% | **NONE (Exact)** |
| **RESNET34** | gcd15 | gcd | 89.33% ± 0.47% | 89.33% ± 0.47% | 91.08% ± 0.23% | 91.08% ± 0.23% | 90.97% ± 0.29% | 90.97% ± 0.29% | **NONE (Exact)** |
| **RESNET34** | joint15 | ccsn | 59.90% ± 2.48% | 59.90% ± 2.48% | 58.12% ± 1.29% | 58.12% ± 1.29% | 57.88% ± 2.14% | 57.88% ± 2.14% | **NONE (Exact)** |
| **RESNET34** | joint15 | gcd | 87.59% ± 2.11% | 87.59% ± 2.11% | 88.70% ± 3.12% | 88.70% ± 3.12% | 88.89% ± 2.94% | 88.89% ± 2.94% | **NONE (Exact)** |
| **RESNET34** | joint15 | joint | 83.69% ± 2.12% | 83.69% ± 2.12% | — | — | — | — | **NONE (Exact)** |
| **RESNET34** | joint15 | source_balanced_average | 73.74% ± 2.20% | 73.74% ± 2.20% | — | — | — | — | **NONE (Exact)** |
| **RESNET50** | ccsn15 | ccsn | 59.12% ± 1.37% | 59.12% ± 1.37% | 57.70% ± 1.77% | 57.70% ± 1.77% | 57.24% ± 1.43% | 57.24% ± 1.43% | **NONE (Exact)** |
| **RESNET50** | ccsn90 | ccsn | 58.55% ± 3.03% | 58.55% ± 3.03% | 54.81% ± 2.70% | 54.81% ± 2.70% | 54.77% ± 2.48% | 54.77% ± 2.48% | **NONE (Exact)** |
| **RESNET50** | gcd15 | gcd | 89.49% ± 0.40% | 89.49% ± 0.40% | 91.45% ± 0.38% | 91.45% ± 0.38% | 91.37% ± 0.13% | 91.37% ± 0.13% | **NONE (Exact)** |
| **RESNET50** | joint15 | ccsn | 61.54% ± 2.94% | 61.54% ± 2.94% | 59.03% ± 1.01% | 59.03% ± 1.01% | 58.89% ± 1.72% | 58.89% ± 1.72% | **NONE (Exact)** |
| **RESNET50** | joint15 | gcd | 86.90% ± 1.04% | 86.90% ± 1.04% | 88.15% ± 1.37% | 88.15% ± 1.37% | 88.72% ± 1.11% | 88.72% ± 1.11% | **NONE (Exact)** |
| **RESNET50** | joint15 | joint | 83.33% ± 1.29% | 83.33% ± 1.29% | — | — | — | — | **NONE (Exact)** |
| **RESNET50** | joint15 | source_balanced_average | 74.22% ± 1.97% | 74.22% ± 1.97% | — | — | — | — | **NONE (Exact)** |

### 6.3 Three-Seed Aggregate Reconciliation (Max Validation Macro-F1 Criterion - Diagnostic)

| Architecture | Arm / Condition | Evaluated On | Reported Top-1 Acc | Recalculated Top-1 Acc | Reported Bal Acc | Recalculated Bal Acc | Reported Macro-F1 | Recalculated Macro-F1 | Discrepancy? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :-: |
| **RESNET18** | ccsn15 | ccsn | 59.62% ± 0.37% | 59.62% ± 0.37% | 56.24% ± 0.88% | 56.24% ± 0.88% | 56.50% ± 0.57% | 56.50% ± 0.57% | **NONE (Exact)** |
| **RESNET18** | ccsn90 | ccsn | 60.82% ± 0.86% | 60.82% ± 0.86% | 55.84% ± 1.65% | 55.84% ± 1.65% | 56.21% ± 1.47% | 56.21% ± 1.47% | **NONE (Exact)** |
| **RESNET18** | gcd15 | gcd | 89.68% ± 0.41% | 89.68% ± 0.41% | 91.68% ± 0.66% | 91.68% ± 0.66% | 91.53% ± 0.31% | 91.53% ± 0.31% | **NONE (Exact)** |
| **RESNET18** | joint15 | ccsn | 62.32% ± 1.42% | 62.32% ± 1.42% | 59.68% ± 1.73% | 59.68% ± 1.73% | 59.81% ± 1.36% | 59.81% ± 1.36% | **NONE (Exact)** |
| **RESNET18** | joint15 | gcd | 88.76% ± 0.30% | 88.76% ± 0.30% | 90.29% ± 0.25% | 90.29% ± 0.25% | 90.47% ± 0.23% | 90.47% ± 0.23% | **NONE (Exact)** |
| **RESNET18** | joint15 | joint | 85.05% ± 0.11% | 85.05% ± 0.11% | — | — | — | — | **NONE (Exact)** |
| **RESNET18** | joint15 | source_balanced_average | 75.54% ± 0.58% | 75.54% ± 0.58% | — | — | — | — | **NONE (Exact)** |
| **RESNET34** | ccsn15 | ccsn | 57.91% ± 1.07% | 57.91% ± 1.07% | 55.86% ± 0.72% | 55.86% ± 0.72% | 55.53% ± 1.10% | 55.53% ± 1.10% | **NONE (Exact)** |
| **RESNET34** | ccsn90 | ccsn | 58.26% ± 1.71% | 58.26% ± 1.71% | 53.56% ± 0.90% | 53.56% ± 0.90% | 53.46% ± 0.62% | 53.46% ± 0.62% | **NONE (Exact)** |
| **RESNET34** | gcd15 | gcd | 89.19% ± 0.53% | 89.19% ± 0.53% | 91.16% ± 0.75% | 91.16% ± 0.75% | 90.95% ± 0.69% | 90.95% ± 0.69% | **NONE (Exact)** |
| **RESNET34** | joint15 | ccsn | 60.97% ± 0.65% | 60.97% ± 0.65% | 58.31% ± 0.91% | 58.31% ± 0.91% | 58.59% ± 0.67% | 58.59% ± 0.67% | **NONE (Exact)** |
| **RESNET34** | joint15 | gcd | 88.89% ± 0.24% | 88.89% ± 0.24% | 90.49% ± 0.34% | 90.49% ± 0.34% | 90.67% ± 0.26% | 90.67% ± 0.26% | **NONE (Exact)** |
| **RESNET34** | joint15 | joint | 84.96% ± 0.13% | 84.96% ± 0.13% | — | — | — | — | **NONE (Exact)** |
| **RESNET34** | joint15 | source_balanced_average | 74.93% ± 0.22% | 74.93% ± 0.22% | — | — | — | — | **NONE (Exact)** |
| **RESNET50** | ccsn15 | ccsn | 61.04% ± 1.01% | 61.04% ± 1.01% | 59.30% ± 1.15% | 59.30% ± 1.15% | 59.10% ± 1.31% | 59.10% ± 1.31% | **NONE (Exact)** |
| **RESNET50** | ccsn90 | ccsn | 60.47% ± 2.89% | 60.47% ± 2.89% | 57.88% ± 1.30% | 57.88% ± 1.30% | 57.73% ± 2.06% | 57.73% ± 2.06% | **NONE (Exact)** |
| **RESNET50** | gcd15 | gcd | 89.73% ± 0.68% | 89.73% ± 0.68% | 91.71% ± 0.67% | 91.71% ± 0.67% | 91.61% ± 0.48% | 91.61% ± 0.48% | **NONE (Exact)** |
| **RESNET50** | joint15 | ccsn | 61.90% ± 1.63% | 61.90% ± 1.63% | 57.43% ± 1.03% | 57.43% ± 1.03% | 57.59% ± 1.06% | 57.59% ± 1.06% | **NONE (Exact)** |
| **RESNET50** | joint15 | gcd | 89.66% ± 0.28% | 89.66% ± 0.28% | 91.35% ± 0.26% | 91.35% ± 0.26% | 91.42% ± 0.32% | 91.42% ± 0.32% | **NONE (Exact)** |
| **RESNET50** | joint15 | joint | 85.76% ± 0.47% | 85.76% ± 0.47% | — | — | — | — | **NONE (Exact)** |
| **RESNET50** | joint15 | source_balanced_average | 75.78% ± 0.96% | 75.78% ± 0.96% | — | — | — | — | **NONE (Exact)** |

### 6.4 Raw Per-Seed Test Metrics Reconciliation (All 36 Production Runs)

Direct comparison of per-seed test metrics recorded in `artifacts/harmonized_thorough/{arch}/{arm}_seed{seed}/harmonized_summary_{arch}.json`.

#### Min Validation Loss Criterion (Canonical)

| Arch | Arm | Seed | Evaluated On | Top-1 Acc (%) | Balanced Acc (%) | Macro-F1 (%) | Selected Epoch | Match Status |
| :--- | :--- | :-: | :--- | :---: | :---: | :---: | :-: | :-: |
| resnet18 | ccsn15 | 42 | CCSN Holdout | 59.83 | 55.46 | 56.46 | 7 | EXACT MATCH |
| resnet18 | ccsn15 | 43 | CCSN Holdout | 57.91 | 54.77 | 54.69 | 12 | EXACT MATCH |
| resnet18 | ccsn15 | 44 | CCSN Holdout | 58.97 | 54.55 | 54.69 | 4 | EXACT MATCH |
| resnet18 | ccsn90 | 42 | CCSN Holdout | 60.68 | 56.77 | 57.01 | 36 | EXACT MATCH |
| resnet18 | ccsn90 | 43 | CCSN Holdout | 61.11 | 57.27 | 57.78 | 42 | EXACT MATCH |
| resnet18 | ccsn90 | 44 | CCSN Holdout | 61.75 | 56.97 | 57.74 | 51 | EXACT MATCH |
| resnet18 | gcd15 | 42 | GCD Holdout | 89.69 | 91.77 | 91.59 | 15 | EXACT MATCH |
| resnet18 | gcd15 | 43 | GCD Holdout | 90.08 | 92.28 | 91.80 | 14 | EXACT MATCH |
| resnet18 | gcd15 | 44 | GCD Holdout | 89.87 | 92.25 | 91.83 | 13 | EXACT MATCH |
| resnet18 | joint15 | 42 | CCSN Holdout | 61.54 | 57.59 | 58.01 | 15 | EXACT MATCH |
| resnet18 | joint15 | 42 | GCD Holdout | 88.75 | 90.06 | 90.45 | 15 | EXACT MATCH |
| resnet18 | joint15 | 42 | Combined Holdout | 84.92 | — | — | 15 | EXACT MATCH |
| resnet18 | joint15 | 42 | Source-Balanced Avg | 75.14 | — | — | 15 | EXACT MATCH |
| resnet18 | joint15 | 43 | CCSN Holdout | 60.90 | 57.52 | 58.00 | 6 | EXACT MATCH |
| resnet18 | joint15 | 43 | GCD Holdout | 87.63 | 88.64 | 89.21 | 6 | EXACT MATCH |
| resnet18 | joint15 | 43 | Combined Holdout | 83.87 | — | — | 6 | EXACT MATCH |
| resnet18 | joint15 | 43 | Source-Balanced Avg | 74.26 | — | — | 6 | EXACT MATCH |
| resnet18 | joint15 | 44 | CCSN Holdout | 60.47 | 58.09 | 58.06 | 4 | EXACT MATCH |
| resnet18 | joint15 | 44 | GCD Holdout | 85.29 | 86.35 | 86.88 | 4 | EXACT MATCH |
| resnet18 | joint15 | 44 | Combined Holdout | 81.80 | — | — | 4 | EXACT MATCH |
| resnet18 | joint15 | 44 | Source-Balanced Avg | 72.88 | — | — | 4 | EXACT MATCH |
| resnet34 | ccsn15 | 42 | CCSN Holdout | 57.48 | 56.48 | 55.49 | 7 | EXACT MATCH |
| resnet34 | ccsn15 | 43 | CCSN Holdout | 56.62 | 55.26 | 54.40 | 3 | EXACT MATCH |
| resnet34 | ccsn15 | 44 | CCSN Holdout | 60.04 | 55.47 | 56.15 | 4 | EXACT MATCH |
| resnet34 | ccsn90 | 42 | CCSN Holdout | 57.26 | 55.76 | 55.01 | 7 | EXACT MATCH |
| resnet34 | ccsn90 | 43 | CCSN Holdout | 55.98 | 55.23 | 54.22 | 3 | EXACT MATCH |
| resnet34 | ccsn90 | 44 | CCSN Holdout | 59.19 | 54.38 | 55.14 | 4 | EXACT MATCH |
| resnet34 | gcd15 | 42 | GCD Holdout | 89.13 | 90.97 | 91.08 | 15 | EXACT MATCH |
| resnet34 | gcd15 | 43 | GCD Holdout | 89.87 | 91.35 | 91.20 | 9 | EXACT MATCH |
| resnet34 | gcd15 | 44 | GCD Holdout | 88.99 | 90.93 | 90.64 | 12 | EXACT MATCH |
| resnet34 | joint15 | 42 | CCSN Holdout | 60.26 | 58.34 | 58.40 | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | GCD Holdout | 89.17 | 90.86 | 90.97 | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | Combined Holdout | 85.11 | — | — | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | Source-Balanced Avg | 74.71 | — | — | 15 | EXACT MATCH |
| resnet34 | joint15 | 43 | CCSN Holdout | 57.26 | 56.73 | 55.53 | 3 | EXACT MATCH |
| resnet34 | joint15 | 43 | GCD Holdout | 85.19 | 85.13 | 85.53 | 3 | EXACT MATCH |
| resnet34 | joint15 | 43 | Combined Holdout | 81.26 | — | — | 3 | EXACT MATCH |
| resnet34 | joint15 | 43 | Source-Balanced Avg | 71.23 | — | — | 3 | EXACT MATCH |
| resnet34 | joint15 | 44 | CCSN Holdout | 62.18 | 59.29 | 59.71 | 11 | EXACT MATCH |
| resnet34 | joint15 | 44 | GCD Holdout | 88.40 | 90.12 | 90.18 | 11 | EXACT MATCH |
| resnet34 | joint15 | 44 | Combined Holdout | 84.71 | — | — | 11 | EXACT MATCH |
| resnet34 | joint15 | 44 | Source-Balanced Avg | 75.29 | — | — | 11 | EXACT MATCH |
| resnet50 | ccsn15 | 42 | CCSN Holdout | 58.12 | 56.21 | 55.82 | 3 | EXACT MATCH |
| resnet50 | ccsn15 | 43 | CCSN Holdout | 60.68 | 59.66 | 58.67 | 5 | EXACT MATCH |
| resnet50 | ccsn15 | 44 | CCSN Holdout | 58.55 | 57.24 | 57.24 | 4 | EXACT MATCH |
| resnet50 | ccsn90 | 42 | CCSN Holdout | 56.20 | 51.69 | 52.00 | 2 | EXACT MATCH |
| resnet50 | ccsn90 | 43 | CCSN Holdout | 57.48 | 56.45 | 55.54 | 5 | EXACT MATCH |
| resnet50 | ccsn90 | 44 | CCSN Holdout | 61.97 | 56.29 | 56.77 | 4 | EXACT MATCH |
| resnet50 | gcd15 | 42 | GCD Holdout | 89.76 | 91.76 | 91.47 | 10 | EXACT MATCH |
| resnet50 | gcd15 | 43 | GCD Holdout | 89.03 | 91.02 | 91.22 | 9 | EXACT MATCH |
| resnet50 | gcd15 | 44 | GCD Holdout | 89.69 | 91.57 | 91.41 | 9 | EXACT MATCH |
| resnet50 | joint15 | 42 | CCSN Holdout | 58.97 | 58.99 | 57.64 | 2 | EXACT MATCH |
| resnet50 | joint15 | 42 | GCD Holdout | 86.20 | 87.49 | 87.99 | 2 | EXACT MATCH |
| resnet50 | joint15 | 42 | Combined Holdout | 82.37 | — | — | 2 | EXACT MATCH |
| resnet50 | joint15 | 42 | Source-Balanced Avg | 72.59 | — | — | 2 | EXACT MATCH |
| resnet50 | joint15 | 43 | CCSN Holdout | 64.74 | 60.06 | 60.85 | 7 | EXACT MATCH |
| resnet50 | joint15 | 43 | GCD Holdout | 88.09 | 89.73 | 90.00 | 7 | EXACT MATCH |
| resnet50 | joint15 | 43 | Combined Holdout | 84.80 | — | — | 7 | EXACT MATCH |
| resnet50 | joint15 | 43 | Source-Balanced Avg | 76.41 | — | — | 7 | EXACT MATCH |
| resnet50 | joint15 | 44 | CCSN Holdout | 60.90 | 58.04 | 58.17 | 4 | EXACT MATCH |
| resnet50 | joint15 | 44 | GCD Holdout | 86.41 | 87.23 | 88.17 | 4 | EXACT MATCH |
| resnet50 | joint15 | 44 | Combined Holdout | 82.82 | — | — | 4 | EXACT MATCH |
| resnet50 | joint15 | 44 | Source-Balanced Avg | 73.65 | — | — | 4 | EXACT MATCH |

#### Max Validation Macro-F1 Criterion (Diagnostic)

| Arch | Arm | Seed | Evaluated On | Top-1 Acc (%) | Balanced Acc (%) | Macro-F1 (%) | Selected Epoch | Match Status |
| :--- | :--- | :-: | :--- | :---: | :---: | :---: | :-: | :-: |
| resnet18 | ccsn15 | 42 | CCSN Holdout | 59.83 | 55.46 | 56.46 | 7 | EXACT MATCH |
| resnet18 | ccsn15 | 43 | CCSN Holdout | 59.19 | 56.08 | 55.96 | 10 | EXACT MATCH |
| resnet18 | ccsn15 | 44 | CCSN Holdout | 59.83 | 57.19 | 57.09 | 10 | EXACT MATCH |
| resnet18 | ccsn90 | 42 | CCSN Holdout | 60.68 | 56.77 | 57.01 | 36 | EXACT MATCH |
| resnet18 | ccsn90 | 43 | CCSN Holdout | 60.04 | 53.93 | 54.51 | 64 | EXACT MATCH |
| resnet18 | ccsn90 | 44 | CCSN Holdout | 61.75 | 56.82 | 57.10 | 36 | EXACT MATCH |
| resnet18 | gcd15 | 42 | GCD Holdout | 89.69 | 91.77 | 91.59 | 15 | EXACT MATCH |
| resnet18 | gcd15 | 43 | GCD Holdout | 90.08 | 92.28 | 91.80 | 14 | EXACT MATCH |
| resnet18 | gcd15 | 44 | GCD Holdout | 89.27 | 90.98 | 91.19 | 8 | EXACT MATCH |
| resnet18 | joint15 | 42 | CCSN Holdout | 63.03 | 60.21 | 60.26 | 12 | EXACT MATCH |
| resnet18 | joint15 | 42 | GCD Holdout | 88.75 | 90.30 | 90.54 | 12 | EXACT MATCH |
| resnet18 | joint15 | 42 | Combined Holdout | 85.14 | — | — | 12 | EXACT MATCH |
| resnet18 | joint15 | 42 | Source-Balanced Avg | 75.89 | — | — | 12 | EXACT MATCH |
| resnet18 | joint15 | 43 | CCSN Holdout | 60.68 | 57.75 | 58.29 | 14 | EXACT MATCH |
| resnet18 | joint15 | 43 | GCD Holdout | 89.06 | 90.54 | 90.65 | 14 | EXACT MATCH |
| resnet18 | joint15 | 43 | Combined Holdout | 85.08 | — | — | 14 | EXACT MATCH |
| resnet18 | joint15 | 43 | Source-Balanced Avg | 74.87 | — | — | 14 | EXACT MATCH |
| resnet18 | joint15 | 44 | CCSN Holdout | 63.25 | 61.09 | 60.89 | 12 | EXACT MATCH |
| resnet18 | joint15 | 44 | GCD Holdout | 88.47 | 90.04 | 90.21 | 12 | EXACT MATCH |
| resnet18 | joint15 | 44 | Combined Holdout | 84.92 | — | — | 12 | EXACT MATCH |
| resnet18 | joint15 | 44 | Source-Balanced Avg | 75.86 | — | — | 12 | EXACT MATCH |
| resnet34 | ccsn15 | 42 | CCSN Holdout | 56.84 | 55.03 | 54.34 | 9 | EXACT MATCH |
| resnet34 | ccsn15 | 43 | CCSN Holdout | 57.91 | 56.30 | 55.76 | 10 | EXACT MATCH |
| resnet34 | ccsn15 | 44 | CCSN Holdout | 58.97 | 56.24 | 56.50 | 7 | EXACT MATCH |
| resnet34 | ccsn90 | 42 | CCSN Holdout | 58.12 | 52.75 | 52.75 | 61 | EXACT MATCH |
| resnet34 | ccsn90 | 43 | CCSN Holdout | 56.62 | 54.53 | 53.92 | 7 | EXACT MATCH |
| resnet34 | ccsn90 | 44 | CCSN Holdout | 60.04 | 53.41 | 53.71 | 44 | EXACT MATCH |
| resnet34 | gcd15 | 42 | GCD Holdout | 89.17 | 91.59 | 91.22 | 11 | EXACT MATCH |
| resnet34 | gcd15 | 43 | GCD Holdout | 88.68 | 90.30 | 90.16 | 7 | EXACT MATCH |
| resnet34 | gcd15 | 44 | GCD Holdout | 89.73 | 91.60 | 91.46 | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | CCSN Holdout | 60.26 | 58.34 | 58.40 | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | GCD Holdout | 89.17 | 90.86 | 90.97 | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | Combined Holdout | 85.11 | — | — | 15 | EXACT MATCH |
| resnet34 | joint15 | 42 | Source-Balanced Avg | 74.71 | — | — | 15 | EXACT MATCH |
| resnet34 | joint15 | 43 | CCSN Holdout | 61.11 | 57.39 | 58.04 | 14 | EXACT MATCH |
| resnet34 | joint15 | 43 | GCD Holdout | 88.75 | 90.43 | 90.51 | 14 | EXACT MATCH |
| resnet34 | joint15 | 43 | Combined Holdout | 84.86 | — | — | 14 | EXACT MATCH |
| resnet34 | joint15 | 43 | Source-Balanced Avg | 74.93 | — | — | 14 | EXACT MATCH |
| resnet34 | joint15 | 44 | CCSN Holdout | 61.54 | 59.21 | 59.34 | 15 | EXACT MATCH |
| resnet34 | joint15 | 44 | GCD Holdout | 88.75 | 90.18 | 90.54 | 15 | EXACT MATCH |
| resnet34 | joint15 | 44 | Combined Holdout | 84.92 | — | — | 15 | EXACT MATCH |
| resnet34 | joint15 | 44 | Source-Balanced Avg | 75.14 | — | — | 15 | EXACT MATCH |
| resnet50 | ccsn15 | 42 | CCSN Holdout | 62.18 | 60.23 | 60.57 | 10 | EXACT MATCH |
| resnet50 | ccsn15 | 43 | CCSN Holdout | 60.68 | 59.66 | 58.67 | 5 | EXACT MATCH |
| resnet50 | ccsn15 | 44 | CCSN Holdout | 60.26 | 58.02 | 58.07 | 8 | EXACT MATCH |
| resnet50 | ccsn90 | 42 | CCSN Holdout | 63.25 | 59.00 | 59.62 | 41 | EXACT MATCH |
| resnet50 | ccsn90 | 43 | CCSN Holdout | 57.48 | 56.45 | 55.54 | 5 | EXACT MATCH |
| resnet50 | ccsn90 | 44 | CCSN Holdout | 60.68 | 58.18 | 58.02 | 21 | EXACT MATCH |
| resnet50 | gcd15 | 42 | GCD Holdout | 89.76 | 91.76 | 91.47 | 10 | EXACT MATCH |
| resnet50 | gcd15 | 43 | GCD Holdout | 89.03 | 91.02 | 91.22 | 9 | EXACT MATCH |
| resnet50 | gcd15 | 44 | GCD Holdout | 90.39 | 92.35 | 92.15 | 15 | EXACT MATCH |
| resnet50 | joint15 | 42 | CCSN Holdout | 60.47 | 56.69 | 56.85 | 14 | EXACT MATCH |
| resnet50 | joint15 | 42 | GCD Holdout | 89.38 | 91.06 | 91.09 | 14 | EXACT MATCH |
| resnet50 | joint15 | 42 | Combined Holdout | 85.32 | — | — | 14 | EXACT MATCH |
| resnet50 | joint15 | 42 | Source-Balanced Avg | 74.92 | — | — | 14 | EXACT MATCH |
| resnet50 | joint15 | 43 | CCSN Holdout | 63.68 | 56.99 | 57.12 | 14 | EXACT MATCH |
| resnet50 | joint15 | 43 | GCD Holdout | 89.94 | 91.57 | 91.72 | 14 | EXACT MATCH |
| resnet50 | joint15 | 43 | Combined Holdout | 86.25 | — | — | 14 | EXACT MATCH |
| resnet50 | joint15 | 43 | Source-Balanced Avg | 76.81 | — | — | 14 | EXACT MATCH |
| resnet50 | joint15 | 44 | CCSN Holdout | 61.54 | 58.60 | 58.80 | 12 | EXACT MATCH |
| resnet50 | joint15 | 44 | GCD Holdout | 89.66 | 91.41 | 91.45 | 12 | EXACT MATCH |
| resnet50 | joint15 | 44 | Combined Holdout | 85.71 | — | — | 12 | EXACT MATCH |
| resnet50 | joint15 | 44 | Source-Balanced Avg | 75.60 | — | — | 12 | EXACT MATCH |