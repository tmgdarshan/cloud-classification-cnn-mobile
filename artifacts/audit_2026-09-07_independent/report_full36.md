# Full 36-Run Harmonized Matrix Independent Audit

- Branch `feature/independent-pool-tuning`, HEAD `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`, `origin/main` `15e6df57b5b8eb2b5c54cfe6714d9381978312e9`, requested baseline `15e6df57`.
- CUDA: `True` on `NVIDIA GeForce RTX 4070`; no CPU fallback is permitted by `src/run_harmonized.py:541-542`.
- Inventory completeness: 36/36 runs complete with summary, two checkpoints, and two metadata sidecars.

## Hash Integrity
Computed independent SHA-256 hashes for 210 expected files. Compared 208 hashes parsed from Antigravity's inventory.
- Inventory hash mismatches: 0
- Distinct files genuinely sharing a hash: 0
- Checkpoints sharing a hash across runs: 0
No Antigravity-reported hash differed from the independently computed hash for parsed rows.

### Computed Hashes
| Category | Path | Bytes | Computed SHA-256 |
| --- | --- | --- | --- |
| master_compiled_file | artifacts/harmonized_thorough/ARTIFACT_INVENTORY.md | 92497 | 18f61afdc8935d4b7d9b23fa177acc4e4fb89080a5ddebea32835a359cd7ed6d |
| master_compiled_file | artifacts/harmonized_thorough/compute_environment.json | 281 | 30cfcc41767f56f6291718430173a9949b0eca30f39658b9317a1b50618b4f51 |
| master_compiled_file | artifacts/harmonized_thorough/full_pipeline.log | 7834 | 04a580289034700df3e8dcd23d1dfd3c60d007ab3ee302df6b65e2add4f0ccf9 |
| master_compiled_file | artifacts/harmonized_thorough/master_multi_seed_report.md | 10651 | 4f7d3fb05e2f7643a603b68d2106fa3b4a3a2fc630ba7303cccbb61b0ac5b633 |
| master_compiled_file | artifacts/harmonized_thorough/master_multi_seed_results.json | 525629 | e102098f2ee9ee0ed01eff1c4d1119ac2d09861006eec00f284c519a62350b9e |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed42/ccsn_model_resnet18.meta.json | 974 | 5473a31c8e2fc799b7afa6388128a107eb6673ff501cc5b987e1bd927e303620 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed42/ccsn_model_resnet18.pth | 45323308 | 8662a5dbe34f3a898531d681fe1f6421f9c9315d60f35a4315d6ee10a0705813 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed42/ccsn_model_resnet18_best_f1.meta.json | 986 | ced98ed8b179bf29b482920f1444f43f3b125f0d7e96e88e47e6c28380a239ba |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed42/ccsn_model_resnet18_best_f1.pth | 45324436 | d9bdade3efd9bc58664e15953c092d8cb2d032fe72e4b8e453bdfd5e56ba03a2 |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed42/harmonized_summary_resnet18.json | 16353 | fcd586cc2a2fc8051e75116035a9fbcd7a0c0409b586be0e0f639ed0f538b545 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed43/ccsn_model_resnet18.meta.json | 975 | e282d971178521ef1362566667546d1f773737b9e9c0d057ddffdd194964aa03 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed43/ccsn_model_resnet18.pth | 45323308 | d31d6f5822ff5c6c5a33036d93b1614655d1b92a4a02c8e77d94a46f1cb89d46 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed43/ccsn_model_resnet18_best_f1.meta.json | 988 | 39f5ef49892b65f25d88fb4f9bbb2ee6f68557dd2072fb90e25793c959c1c4d7 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed43/ccsn_model_resnet18_best_f1.pth | 45324436 | ce2907a659c1009d9e2cd1ebc0ead960684ce3046b99b90b3818f10e0bdc323c |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed43/harmonized_summary_resnet18.json | 16357 | b51d631049130ba253cb6baa2315c6fa336d03dfafcb46f49e6b4d88e956ec9e |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed44/ccsn_model_resnet18.meta.json | 974 | a60540a95a69a56809f7df9e1acd91f19f4e35a1bd1c153b88ed72eed6a6f71c |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed44/ccsn_model_resnet18.pth | 45323308 | 53efd23a6001f4690155c2625b13c7977a840c00c771556d67c372cb82086d56 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed44/ccsn_model_resnet18_best_f1.meta.json | 988 | ed12a153da3a3daffca504460f5f72500e12962da2dc2c0aba9b420d93700fde |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn15_seed44/ccsn_model_resnet18_best_f1.pth | 45324436 | c80146b2d2e6bc3b3a9b0daf5503aecb01dc88c039dbfe3859511824b4d9860b |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn15_seed44/harmonized_summary_resnet18.json | 16357 | 2eca0f510958e221e53a7fb7c3e9acae29ea4887e62acc2b41dcda2a2fe5331e |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed42/ccsn_model_resnet18.meta.json | 976 | 47a2f1529b620850b1a8025d13938d5509f92cb062ce3f7a235921b0d302710c |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed42/ccsn_model_resnet18.pth | 45323308 | 3f8b0fec5303af3fd7186ffc71fc3fe5fb60907abe4b3cc7dfc38166ab382bd4 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed42/ccsn_model_resnet18_best_f1.meta.json | 988 | ecb2b230b663a6a51f341b29c723a1b09730db786461386d99e0864aca9e2b22 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed42/ccsn_model_resnet18_best_f1.pth | 45324436 | 8fe78c93d9747e4851a9bcf31205f51675173f22b07f9b42fe566ecc881a605e |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed42/harmonized_summary_resnet18.json | 16356 | 335cc1e6f9c2ff49d46222e2ef24f5cd266cfb630d3b7d09156cbbe48945b630 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed43/ccsn_model_resnet18.meta.json | 975 | 7e2cdb5c1449552868bc42a8656f233088bbb74a4a434bf739a3b404f2197500 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed43/ccsn_model_resnet18.pth | 45323308 | b6b15063b77ba37d2c19539a9f4cf89de5300ca97775daf8a3e3379d07507ea1 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed43/ccsn_model_resnet18_best_f1.meta.json | 987 | 2f4d525ca78f035e4c5df81f43716d38ccad5f233300692a1e8461098e7143a6 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed43/ccsn_model_resnet18_best_f1.pth | 45324436 | d94bf3e25f167b2965a5aa9e447c21732193311a89b63765aac2f99171ed5c83 |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed43/harmonized_summary_resnet18.json | 16354 | 718a4bf40fa0222f0cf3b9e167ee5987dcda669e410fd0a16b4d6bcf85a3a738 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed44/ccsn_model_resnet18.meta.json | 975 | d69f6573aa6a19fab80c2489c42a37cd3528f2ceff084ec7848280659dd5f6ac |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed44/ccsn_model_resnet18.pth | 45323308 | bbbdaa68d81a36057519bad063e4b7187a5d7b4bc4120d9acf145e076e26c140 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed44/ccsn_model_resnet18_best_f1.meta.json | 988 | 830b1cc175d6dc1d962664fb1e487f3ca54b9cc64cd55cb3c405a15b29f9474f |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/ccsn90_seed44/ccsn_model_resnet18_best_f1.pth | 45324436 | 165409140d9b9458796e8265a99bf6d0d5e2780be8ffa186af624389f576ab14 |
| summary_json | artifacts/harmonized_thorough/resnet18/ccsn90_seed44/harmonized_summary_resnet18.json | 16349 | 8ba565ef0c084070d8a7f9b122a47809709d58ca5696a08035ed16af11c4aa86 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed42/gcd_model_resnet18.meta.json | 974 | 0d407e73c0a5af411fc2bb9a6610a5c5b00c6a7ae0e4e93e0733b2bf6e26f5b9 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed42/gcd_model_resnet18.pth | 45323175 | 8be3453b34c972511253ba7e06c289d59a60ccaecd263403fcb3868efbd45bc9 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed42/gcd_model_resnet18_best_f1.meta.json | 986 | 78f86c76190dc8128793a0779eeaf14db52b25d7f877204ac3849c8b9c8fcdb2 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed42/gcd_model_resnet18_best_f1.pth | 45324303 | ac815b49efd5a33295f2654b013304ae122d55a3d10643ea7f54dedf46c73e25 |
| summary_json | artifacts/harmonized_thorough/resnet18/gcd15_seed42/harmonized_summary_resnet18.json | 16347 | 51c9b7e9f328ae3535bfdd9813368542f4925e10c2f7598ab70f9bb515f20067 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed43/gcd_model_resnet18.meta.json | 974 | 5354c96ddc305529220218cc8d19d9ff03a6cb06bcc52949b479261aaf799f1b |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed43/gcd_model_resnet18.pth | 45323175 | 9c69a9cd330ae8872910ebc8b7a12b5472128c564514c237c96b819eeec18416 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed43/gcd_model_resnet18_best_f1.meta.json | 986 | 1b547201b928f18ea50b13de1ec1aacf54e4372cbdc29032dc72c1b50d28fd15 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed43/gcd_model_resnet18_best_f1.pth | 45324303 | e8381d0f683705e0b0f98a13b73b20d04a15ef643cb34053b50266c22fc41ca7 |
| summary_json | artifacts/harmonized_thorough/resnet18/gcd15_seed43/harmonized_summary_resnet18.json | 16338 | 2c2f77a2700c060fa7c3e039ab15a43e25b65d035d72ee935b68bdb7082f0fd1 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed44/gcd_model_resnet18.meta.json | 974 | ca10e0a963d75fe324e81f7202fbb4af1341bfaeec4b70432d6b902d8cb6e64c |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed44/gcd_model_resnet18.pth | 45323175 | 935ee55ad7e2b3d920c1876103a4b0b0a74174a554dc2e42d2930462872576d0 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/gcd15_seed44/gcd_model_resnet18_best_f1.meta.json | 984 | 9e8750c22002a6c51fc6f8c49ce4253bbd2a76bc842b61a69d1a8d012f8e136e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/gcd15_seed44/gcd_model_resnet18_best_f1.pth | 45324303 | 969095790b39df128e09fc9045b5eb2b87bdc4135c481e01dc5b585559bb4381 |
| summary_json | artifacts/harmonized_thorough/resnet18/gcd15_seed44/harmonized_summary_resnet18.json | 16335 | b6cd728c1579bdd5aab49233a0da73e120fe72a9e8bb6c647aaf9cf6691692a7 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed42/harmonized_joint_resnet18.meta.json | 979 | e161e91a98a0da7a733388949bde9ebfeca6ff25a20bee7783705f2a4be3269b |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed42/harmonized_joint_resnet18.pth | 45324170 | 4beeac7b4a70c8896d86c21bade82872f95790fcc150284b16666940dfa35b9c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed42/harmonized_joint_resnet18_best_f1.meta.json | 991 | b2e6aec63483aea5578b93df14cfeb2a62af81274475132d2123846a0111520e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed42/harmonized_joint_resnet18_best_f1.pth | 45325234 | 2d78a39ddb4aa57e82dbfa669d1a1d02a7718b092551f7d59ad726f0f462261d |
| summary_json | artifacts/harmonized_thorough/resnet18/joint_seed42/harmonized_summary_resnet18.json | 22648 | 45fc537dec746da1c4bb57aa9fcd5fdbd7d722dae02aabc28722887c65546298 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed43/harmonized_joint_resnet18.meta.json | 975 | 8c3b7fb0365b92076a6057ee864e8e187adafebc9aa971ff82da3ffff627f582 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed43/harmonized_joint_resnet18.pth | 45324170 | 66d4c9414f4d0259bfe9327e06fe5917973e2fb93678ea784cc9915a22af45b6 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed43/harmonized_joint_resnet18_best_f1.meta.json | 991 | 280b0377a2340569c2410c438853d43cbdf8d400daf79b84ea424eb68e1c0d73 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed43/harmonized_joint_resnet18_best_f1.pth | 45325234 | 684f7974eaef53903ff8d2b13fc0fab94fa1e88ad43ca89ea80bfcc12c0b26be |
| summary_json | artifacts/harmonized_thorough/resnet18/joint_seed43/harmonized_summary_resnet18.json | 22648 | 4ffa3452d1c572d370f2ec115319952c36ede73bd95eedd7bd4dc9cb1a7bc521 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed44/harmonized_joint_resnet18.meta.json | 977 | 82853c1d0d46e1f22b30d944e12f71bed83fd3202720ce7fe55eb9e072aea5ee |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed44/harmonized_joint_resnet18.pth | 45324170 | 71ffd07428a407263d03ba38aae487143c7f1cb067e96a45c9f13154ef0935ce |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet18/joint_seed44/harmonized_joint_resnet18_best_f1.meta.json | 991 | a64e1b63f1d75c42a21990231776b59b3571c345d64f0a9a6f9ce73c9ee1a465 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet18/joint_seed44/harmonized_joint_resnet18_best_f1.pth | 45325234 | e71a18e5ffb440ea2e488fb75a20de1646320219b6a340455d17e07ad4cd55c4 |
| summary_json | artifacts/harmonized_thorough/resnet18/joint_seed44/harmonized_summary_resnet18.json | 22645 | 3c89136c611a6893c018f717086e95b4c7515e5a59738042d661099ba4ed33ac |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed42/ccsn_model_resnet34.meta.json | 973 | 0190789bfa23c3f8daacf34447eced4153726c2355dc0a9de4cd7f0205a4ab6f |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed42/ccsn_model_resnet34.pth | 85818348 | d7cb7f64fb5e865fa728166396fe285b4b4c9c5bf872d8c6b6a9ec04fda70a83 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed42/ccsn_model_resnet34_best_f1.meta.json | 986 | 9f4f2fd3e23a09a889214784514351b23525cc4ff0cfb0771786d5a9dfd8ae04 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed42/ccsn_model_resnet34_best_f1.pth | 85820244 | 045c9051fc9596d2c3aea849c8c857a32f812a982bd476bac4ddc21d2aca6ed6 |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed42/harmonized_summary_resnet34.json | 16347 | 8c0c18527c246cbef420e633aee2dde72a2ac410b1b2b2b66bf76b4e7c38586c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed43/ccsn_model_resnet34.meta.json | 973 | 416afe5e031c7202ded149db8e6e3e402ce0ad7b564989a1958ba6de5cb04d3e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed43/ccsn_model_resnet34.pth | 85818348 | fc138b8037896b4e026938b4366a5b6f3615f95e796fa10b74d10362e5988c2a |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed43/ccsn_model_resnet34_best_f1.meta.json | 988 | 307d61f382b4cc68d3b459e8fdbeca4dc80bffe1202b4f387ae39daa6591a81c |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed43/ccsn_model_resnet34_best_f1.pth | 85820244 | 2ffb212a89fb0b4deaf3d0d2b07ab0c3e68b6a18190c267c3ccd3b15340ab92d |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed43/harmonized_summary_resnet34.json | 16341 | 7ea3455047c8e31845f772df430e2fdb7c288bb517db40cc09f21e6e6cfa8de0 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed44/ccsn_model_resnet34.meta.json | 974 | b2c27abda4ca8c53a2285319431bbbb96f29b45e9ce9981de50404615f6081f9 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed44/ccsn_model_resnet34.pth | 85818348 | 337e09cb7ea5272b6b6f4e87e44a2b86f508538fdfe57b1854062a6bd814971c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed44/ccsn_model_resnet34_best_f1.meta.json | 986 | 216453b56344f63145f83a722236ea75a3992dd1bc14b7f8017731679618abbd |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn15_seed44/ccsn_model_resnet34_best_f1.pth | 85820244 | bf5e227bc74abbbfbaef5f147e90ef870beea6c4b3bab0f0efcd69fb2614323c |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn15_seed44/harmonized_summary_resnet34.json | 16342 | a711b4f5333ffa6b1fd6f7c4a6e330902c3f57f505f656dccb813da23edcb12b |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed42/ccsn_model_resnet34.meta.json | 974 | 58196a9af168616827508f05e31f9110014a6850b0b47c4c72a720f2ab637afd |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed42/ccsn_model_resnet34.pth | 85818348 | 65729d073c11ff0dcc04e1b1c693542ffd6095e7833ce58fe29d624a5e8ff8b2 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed42/ccsn_model_resnet34_best_f1.meta.json | 987 | 99d5f1030231f62072b8e09cdafc1531a506b94951e20545cf048b8abe3f4dc6 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed42/ccsn_model_resnet34_best_f1.pth | 85820244 | e5dff15494a5571224c36c713f77ad1c65f9d798dcceb5b00c10db741dacda1b |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed42/harmonized_summary_resnet34.json | 16348 | eed2ccea693001e15fd76ddc3b950168996c13039a3df080ffd0638784c75e7d |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed43/ccsn_model_resnet34.meta.json | 974 | 7ec25bad8c68ab13a44dec0ec6fe391c1398e598fc25cebbf610697d00290211 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed43/ccsn_model_resnet34.pth | 85818348 | 88b1fb31b592cc3fb2582d2f2da21676e9cc6a5d1f2bf34dd9cd3a0e24fecede |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed43/ccsn_model_resnet34_best_f1.meta.json | 986 | e09e29752610ed8b325f13c72df84b0762e0bfc41f968627ecd731ff2e695bf6 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed43/ccsn_model_resnet34_best_f1.pth | 85820244 | 07c9a330063ab4e8b8a77936203f1a93813f51150b50d71fee9160b15332f0f3 |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed43/harmonized_summary_resnet34.json | 16348 | d35f30a9a10b6b9d23b4547fb0261e9c5eefdb0f90ebf833d928764b5924a0d3 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed44/ccsn_model_resnet34.meta.json | 974 | 738971cdc27b317524b26d952c9d719658f7ef4a6368af9f952b8c01b9b90547 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed44/ccsn_model_resnet34.pth | 85818348 | 37d951ab836b2970a1508edc586796a1689ac6f2a296a2c0e2c452e2fe0522f5 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed44/ccsn_model_resnet34_best_f1.meta.json | 987 | 01b19ee89b5ab9590518b2aaed593c3a01f8397ce2494a99a0cd9b8aaa8ceec2 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/ccsn90_seed44/ccsn_model_resnet34_best_f1.pth | 85820244 | 757527e4835460642c09613f8cce375e2c6bb749acf7e2f92986043600329cb1 |
| summary_json | artifacts/harmonized_thorough/resnet34/ccsn90_seed44/harmonized_summary_resnet34.json | 16339 | 12a4149c2b29a9f2688696f98cdd5b71f0498c52bc4c82ea0e28ab7156fd42e1 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed42/gcd_model_resnet34.meta.json | 974 | cf151fe62d8eb1e91aeda442ee26bb683916dd6cdf41890e42ccc39ded8ce03e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed42/gcd_model_resnet34.pth | 85818119 | d2442949b8e86436038a020858db626e65ad517d8dd39747f12c6da80425887c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed42/gcd_model_resnet34_best_f1.meta.json | 985 | 28ff14d36a1c1b8269596ea2a698b0f1390bb1946b7b059be4a20a26914bf9f0 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed42/gcd_model_resnet34_best_f1.pth | 85820015 | 061c2beff83cf3c05d84c7977bf0d9195aadc5c51dbe419e1c479357475d8b13 |
| summary_json | artifacts/harmonized_thorough/resnet34/gcd15_seed42/harmonized_summary_resnet34.json | 16336 | 45391de37da66c37715920eb2ef5a05095c63237808abeea2e232b3d7a39d9f1 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed43/gcd_model_resnet34.meta.json | 972 | 109bc4701b17313750cd603e2ebaf6408cfb03b16406e998b7ed8c65cbf5d9f3 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed43/gcd_model_resnet34.pth | 85818119 | df1407f94c905648ca3fc7e4eb2bbd5f4f56377aadcca3a10c143bc9a1734619 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed43/gcd_model_resnet34_best_f1.meta.json | 984 | 259f1d64df951cfc2fc8083f11d16552e1e6aa236782fc67d237b93538001a42 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed43/gcd_model_resnet34_best_f1.pth | 85820015 | 256d9eaf9a4c4751f491a3d8861e34aedd4cdf4f3dad3217db2cb0a197e97da1 |
| summary_json | artifacts/harmonized_thorough/resnet34/gcd15_seed43/harmonized_summary_resnet34.json | 16327 | c87063a43bb9cd7e702bfa9081b501fa8a7c8190c9f499c1810b27cfa125af9b |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed44/gcd_model_resnet34.meta.json | 974 | 74f15055e12f8ad8cffce7f29334efd7e643a2d022cdbd96a2645352fdfbb61e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed44/gcd_model_resnet34.pth | 85818119 | 818887796ea3f65ede3785a09d88c60321c6dd56093922f32b45f4378886dced |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/gcd15_seed44/gcd_model_resnet34_best_f1.meta.json | 986 | 26696b55bb5e1dbe0cacecbd4425f9f5dfb9e264d294f21192ab05ff7507cde5 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/gcd15_seed44/gcd_model_resnet34_best_f1.pth | 85820015 | 4911bbd20b3714b186d69d4512608ccef74963db692723f7a6d3ccd72a27af31 |
| summary_json | artifacts/harmonized_thorough/resnet34/gcd15_seed44/harmonized_summary_resnet34.json | 16337 | d7f9b9cb0b142d4d3ed4cfa7952e7e82e3b28d8efd7a41d5febef9c0be59ed5d |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed42/harmonized_joint_resnet34.meta.json | 979 | ab9f19efe7b3c121981a0e3040c1f408b648b237719f1a1df97521ea45500581 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed42/harmonized_joint_resnet34.pth | 85819786 | 516cef796269bc20e1b06b93e2b092f59fd967036563df2e75cea52b4939464a |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed42/harmonized_joint_resnet34_best_f1.meta.json | 991 | 84d15836a250dc8cedcd73caea16f2fb3570b903d3850e4cc72ba0bc6b9ca25f |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed42/harmonized_joint_resnet34_best_f1.pth | 85821618 | 76b4b232c99f59feb7205255a066937d03ed2e737a9b4bf1b497cfa95183a003 |
| summary_json | artifacts/harmonized_thorough/resnet34/joint_seed42/harmonized_summary_resnet34.json | 22660 | f36155b0fac85646643a484819595499f44dd066b6a7d35dd1988edcafe26baf |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed43/harmonized_joint_resnet34.meta.json | 977 | a1e792f76905614ce95cb71a6df0ed84909a9c38df39a94984e48c87a80f2092 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed43/harmonized_joint_resnet34.pth | 85819786 | fd4e1d7d82bea9cbcea71abd8ee1a991d7789b90c22c7e1371f7c97e01036f21 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed43/harmonized_joint_resnet34_best_f1.meta.json | 990 | 58501fdc66e6017a2fbd8263160f391d003111e9f556fb09699967a53efe8fe9 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed43/harmonized_joint_resnet34_best_f1.pth | 85821618 | 16f605cd32e088311b9bc7c10d0fed89c4136c4c55117b1b83c966ccd8c52e0b |
| summary_json | artifacts/harmonized_thorough/resnet34/joint_seed43/harmonized_summary_resnet34.json | 22653 | 292e2b7cf5955aa81d1066cfaf6467222d219da914be56df70b3fb2c2e4f5eb0 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed44/harmonized_joint_resnet34.meta.json | 978 | 469a15a5b9801cf23c84679a6aa830ce1eea917cde0925a4b9d564061a196f0c |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed44/harmonized_joint_resnet34.pth | 85819786 | 176983934a021879855b6973c5e963a946368599ff44c5835ac123267b7bdd19 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet34/joint_seed44/harmonized_joint_resnet34_best_f1.meta.json | 991 | 19b639b454be68cfc28e35aea2211029c2f93215549a7e107a86c9fa0fb9b958 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet34/joint_seed44/harmonized_joint_resnet34_best_f1.pth | 85821618 | a96c355c77c9872a21c2e783551cefa9d6cbf2b36d2ff4d0af273ff3499120f1 |
| summary_json | artifacts/harmonized_thorough/resnet34/joint_seed44/harmonized_summary_resnet34.json | 22656 | fe609bd81c9938ad3d28c01bd33f0659626ff1a990c6b067c353652d18a4592c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed42/ccsn_model_resnet50.meta.json | 973 | 7d24e2a7e5fa979d5ae35a9bc7b169ea2072fe4982c6daf645648a5fbb808ccf |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed42/ccsn_model_resnet50.pth | 96463336 | d11f287dd8660a090dd4792d1f036ca3600b9861e9466087c92bfe0c7499f9bf |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed42/ccsn_model_resnet50_best_f1.meta.json | 986 | 3e6be39ad2875670999ecbe9ccaf7d980079ae129e5446146f2b6c2b9aed9f27 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed42/ccsn_model_resnet50_best_f1.pth | 96466112 | 0dee01585dd5af486e6c71e52b9aac75415c1ec1804ed880957d1c8486eb50b9 |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed42/harmonized_summary_resnet50.json | 16340 | 45fc037b2b5b74412bfe81f970f3ca228f8d9fb3b14f0a6ea0de7bef86914dbc |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed43/ccsn_model_resnet50.meta.json | 974 | f451be58b5232b92d87a164b58a249bc9ed4bb42808d20f55ba4c4c02cc8d305 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed43/ccsn_model_resnet50.pth | 96463336 | 68a29011c3262ec6c5b43bea1053f5246ad74b2017bf41f0dc459635656c876c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed43/ccsn_model_resnet50_best_f1.meta.json | 986 | a97616c9dc10f46c0c7fa04985df9584ebef1544238923894869c91a6e3d305e |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed43/ccsn_model_resnet50_best_f1.pth | 96466112 | cc1712d1b151712512000cff84a877e95012381ca2195f6f89fd237b44f4e8d7 |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed43/harmonized_summary_resnet50.json | 16353 | 5f9ea8cdc93ab11c3a06881a1eb41e57029f095a856a87169094ba26a2cda4b3 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed44/ccsn_model_resnet50.meta.json | 974 | f4fb020f7f84054cda8e4312ad11f6462df5ecebf059c15e0eefd87f665d7b39 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed44/ccsn_model_resnet50.pth | 96463336 | 8c2a7c74b5a38c5f7c73f645c02fadb623008ae927602eba1c0dca231aefc4e7 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed44/ccsn_model_resnet50_best_f1.meta.json | 985 | 64276b3ec3a9002458f94de726c71093c44ad1f27ef0a419e26c10769ccda212 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn15_seed44/ccsn_model_resnet50_best_f1.pth | 96466112 | 7bbcc54736dcba848ab0614137ef9f38387ca16d2f8c5705da424e35f5c1017b |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn15_seed44/harmonized_summary_resnet50.json | 16336 | 672783c4d55f0ec1e786343c3fda15b69ff0f1b0f634e9921087f3565d8d0be5 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed42/ccsn_model_resnet50.meta.json | 974 | 7199b323ee3b0a32896b41edd9cea75b75897412bdeadbf2fb2ddc5383069097 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed42/ccsn_model_resnet50.pth | 96463336 | a68d36d0ac70282b28fdb8561dc9b32abfcadf8b775f06f37f81c761b1180c2c |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed42/ccsn_model_resnet50_best_f1.meta.json | 987 | 231683cee72dda7230b61679130fadb1bd0b1149de3be9436d21d98867f22157 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed42/ccsn_model_resnet50_best_f1.pth | 96466112 | b2907f4cc26ea1fcc0fd044e8817284b316651a137fad0ebea9d5bdafbc6cb1c |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed42/harmonized_summary_resnet50.json | 16345 | f6873aa1f0d37e77a52d4f5eba652dd963e2cd90110615f2e36ea0ea98ec4f1a |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed43/ccsn_model_resnet50.meta.json | 974 | ee262b533eec4f44d1c0e6d95f529043d743fba9fe5655f3cba4f1159c56ecee |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed43/ccsn_model_resnet50.pth | 96463336 | 0c751dfa1a263002317f22dd621adbf22b62a5bf63d7f7171e6d74d16ef7cc3a |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed43/ccsn_model_resnet50_best_f1.meta.json | 986 | e178838bb19d1ed2f6dc3c9d61b0b5c353fdfa4d86b31869464c72ab6e086c20 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed43/ccsn_model_resnet50_best_f1.pth | 96466112 | 7a7301c0bad4190b833a3070506a00008965775cdbbe42b7a5411b74df8ee47e |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed43/harmonized_summary_resnet50.json | 16356 | 51cc46cccfa06603e87af05d728b6b72b857d9aa19e03c54f362ae05e212b629 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed44/ccsn_model_resnet50.meta.json | 974 | fff8e04c7dd8f6103f417fd48800282e1211fef3d2f5bb3e9caf37d16e0f0039 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed44/ccsn_model_resnet50.pth | 96463336 | 93b64377ea9f75ae9bf656d425bc53331e0e339cf3dc7832135c480f438a0089 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed44/ccsn_model_resnet50_best_f1.meta.json | 988 | 41185a619b96e83f82c1850af1b2eef2ceec38aaf7e8fe5c9038890295baa433 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/ccsn90_seed44/ccsn_model_resnet50_best_f1.pth | 96466112 | 57c4be7356281be1e5ed0ea0da7adec5286dc80e03fc86af870636ec4baa4aa9 |
| summary_json | artifacts/harmonized_thorough/resnet50/ccsn90_seed44/harmonized_summary_resnet50.json | 16352 | 5246196381a97e64a9120736bf6c3953dfbd7cd09d1f51a4d1bb5cc533e35ec5 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed42/gcd_model_resnet50.meta.json | 974 | 85005608d481f72d11888e8b1b7f55531b2a0828f0ea3009d34f867498510773 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed42/gcd_model_resnet50.pth | 96463005 | 671ab8252d55c574d02de0102a8f93733fdfc6117f2cc7a33e8f44e1eb807250 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed42/gcd_model_resnet50_best_f1.meta.json | 986 | 65356a82bbeb11c66dfce0dff3395c1f3b7d5bddd5f3368eae5c878c494b3223 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed42/gcd_model_resnet50_best_f1.pth | 96465781 | f0dfc096f8b7d3500539e5fd8532a31c062500d7cb3cf551428fc1cbce9225a4 |
| summary_json | artifacts/harmonized_thorough/resnet50/gcd15_seed42/harmonized_summary_resnet50.json | 16332 | 5773f862e574316cd3c1dde274edc8024c122a930bf654145f1c241ba5ae0abc |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed43/gcd_model_resnet50.meta.json | 972 | 69491f476b9da16896301a076d402489602bd751d8b2eedf1cee4ca6d7ee63d0 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed43/gcd_model_resnet50.pth | 96463005 | 83057c98f4a71ae3bf77f5d49bc6dc72d0c893b040ac41a659865671a4e5641a |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed43/gcd_model_resnet50_best_f1.meta.json | 984 | fe2461b043c9f5ddadd6934c7cae6f0671c517d54478d44e43b70bd7f17a3325 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed43/gcd_model_resnet50_best_f1.pth | 96465781 | 7a7734b016211362975157a8a58a06a49289c0731f26cf6f94859c762d70ce2c |
| summary_json | artifacts/harmonized_thorough/resnet50/gcd15_seed43/harmonized_summary_resnet50.json | 16335 | e820306aee56fe34e0cd51108451c1133c4954f25bf0cc4e3b0f23b5817e335f |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed44/gcd_model_resnet50.meta.json | 972 | e1f56d514531c13f6f286482daae54f82753bf0ed11e38d2dd98462a30e13a8f |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed44/gcd_model_resnet50.pth | 96463005 | 8f3fb8c317f1746cf7d85c23b0ba3a8f8136578de1bfa6bac25e95483282835e |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/gcd15_seed44/gcd_model_resnet50_best_f1.meta.json | 986 | e8f91c654a59256f21a359371b5d63131120bc68bdb9f091db27b84b92a4045f |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/gcd15_seed44/gcd_model_resnet50_best_f1.pth | 96465781 | e06c0dedf52823715845d932a253a28e1f876093145779c1d8e1e1984edbee58 |
| summary_json | artifacts/harmonized_thorough/resnet50/gcd15_seed44/harmonized_summary_resnet50.json | 16336 | 4d1fc688696936113a7ca991ad78345b35f3240412b5ebf8151b664b01dc2d28 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed42/harmonized_joint_resnet50.meta.json | 977 | 34d97fb91d50a96bfe5cfefa53cf021d821ebcbdab589372a7656b7fab47256d |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed42/harmonized_joint_resnet50.pth | 96465450 | 2f6420b4184e9ac8a05a7bf27896bc67ca05d7a3709d66d6b01d3822689559ae |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed42/harmonized_joint_resnet50_best_f1.meta.json | 991 | d04f580e2bd73d0e3a7f5c5d20077b2441cb868a58b981f8cfa738e34f9aa393 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed42/harmonized_joint_resnet50_best_f1.pth | 96468098 | 71ac42af96da8f80eebe13ea6b48f9073a00ea9346a1b4eefb467addc1616758 |
| summary_json | artifacts/harmonized_thorough/resnet50/joint_seed42/harmonized_summary_resnet50.json | 22654 | 03223fdaa3151d34405443e87cf3a4176de525a0cca188942ca2c47b017c10f1 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed43/harmonized_joint_resnet50.meta.json | 977 | e5ae910f68e184abf2850942951b749f188e0f1852ca2fee585aaee42f75b209 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed43/harmonized_joint_resnet50.pth | 96465450 | ce9d11ea6c4a721dee701fc1291a1cb5f927c1476d04d00d60fbebf381e4dcdf |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed43/harmonized_joint_resnet50_best_f1.meta.json | 990 | 3a163e8451cfad51d74541313dacc6f2a645b26d1be0877f05cfddffed1bbc9a |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed43/harmonized_joint_resnet50_best_f1.pth | 96468098 | 1407dfef7b02eceaf69e102416dd3e7b42299f7a551969941aee198afc982621 |
| summary_json | artifacts/harmonized_thorough/resnet50/joint_seed43/harmonized_summary_resnet50.json | 22655 | d16e1f52b394527be3a09dc092e3d583762b28e415367761ba2dbedeb664a8da |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed44/harmonized_joint_resnet50.meta.json | 977 | f6d83918e9421ad4754aea04c92cbf85da318347bc5cd252c02a10ada7381697 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed44/harmonized_joint_resnet50.pth | 96465450 | a97fd73d4b90237a0972d2034fc7dcfe1fd4c8529fd81fed1d191e6bd59ed911 |
| checkpoint_meta_json | artifacts/harmonized_thorough/resnet50/joint_seed44/harmonized_joint_resnet50_best_f1.meta.json | 991 | c91575991156ece11edde094bff8f13733b74060031da68780ef8e292fa16bd0 |
| checkpoint_pth | artifacts/harmonized_thorough/resnet50/joint_seed44/harmonized_joint_resnet50_best_f1.pth | 96468098 | e03518c87d9d9b9e8ecb140cc8ace60c58c73f804407d58b3cce46a71791003e |
| summary_json | artifacts/harmonized_thorough/resnet50/joint_seed44/harmonized_summary_resnet50.json | 22653 | b82793b28317aa9edc1ff280c911016da7819697ab842e61f66348f1af9aa421 |
| tuning_json | artifacts/tuning/tuning_results_resnet18.json | 12850 | 9d816de03ffd58fcc36f044880cce0386c72ce0c28f69dc2333d7bb3a2ed66ca |
| tuning_json | artifacts/tuning/tuning_results_resnet18_ccsn.json | 15141 | 24acbcc498667fb2789898b782522a2011d0ffe300231352add9e8b686634695 |
| tuning_json | artifacts/tuning/tuning_results_resnet18_gcd.json | 15165 | 7009332a3fd6c6f216c736474ec05992b2d64b4a5c71ee73a4dfe7d3827ae66f |
| tuning_json | artifacts/tuning/tuning_results_resnet34.json | 12846 | 9f8668e870ffe1516e88269034464ece7e185f786fa1eb3bc99f1382618dc003 |
| tuning_json | artifacts/tuning/tuning_results_resnet34_ccsn.json | 15156 | 7e77deffa5d43d7326610af5379394f748b9a077d915f879d18df9c5e0d2c2d4 |
| tuning_json | artifacts/tuning/tuning_results_resnet34_gcd.json | 15160 | e0a4b98b2e73c4a6126b49a9853b91a4eb307b4eb80ebe2ec76075ffb639d9f0 |
| tuning_json | artifacts/tuning/tuning_results_resnet50.json | 12840 | fb291ef02d564f9d86258d4b8710a72681646bbde2ee6bd49ae2c398a171faff |
| tuning_json | artifacts/tuning/tuning_results_resnet50_ccsn.json | 15165 | 70a015e4f05cfce7f8a9220dd9fd95e93bb834cf221899f89ca372e2222b4f42 |
| tuning_json | artifacts/tuning/tuning_results_resnet50_gcd.json | 15151 | 23dcc366fab41f3e699bac0e00027b5dbd4f3814e9780d2bee7444916e741c8a |
| config_toml | config/training/baseline.toml | 411 | 917390e6045d2c5285e6db228ce52890afec9dbb4d896d895db99ebb0921a7f6 |
| config_toml | config/training/smoke.toml | 258 | 941887b938ab07a570dd887684a7bf71f939d9f87cc23b9b32e833f502b1945e |
| config_toml | config/training/tuned_resnet18.toml | 781 | f8c8021475f5691004aa6fab60078e4f219bc9b360245578f5669889ae4e7e64 |
| config_toml | config/training/tuned_resnet18_ccsn.toml | 1200 | 671005513539d544e206d8d4f3428c5b8e4a4a02a6aeeb1f483dcd8a6f67f7e0 |
| config_toml | config/training/tuned_resnet18_ccsn_15ep.toml | 1156 | 5767f5e948fea12f32261f1459eb3dd753198dce4eef5af86b6c0e1b8423acd2 |
| config_toml | config/training/tuned_resnet18_ccsn_budget_matched.toml | 2889 | 54291b51632e58f91a08cb2eff553247711a8973521b0d99031f6af0f43d9876 |
| config_toml | config/training/tuned_resnet18_gcd.toml | 917 | b333f5476eb72a1c1b38bd9d79474beb7b823f63d6650ed71d5236436868e1e3 |
| config_toml | config/training/tuned_resnet18_joint.toml | 800 | 5b5acc797d6b12d899638e9c4bc4a5f8617ca52eb9a7f7e5b41a6b8b62a852b0 |
| config_toml | config/training/tuned_resnet34.toml | 781 | 3e9364e85e6588413ba3986c648483efcc0c2874a6a3e61dee4350aa6aa44287 |
| config_toml | config/training/tuned_resnet34_ccsn.toml | 1208 | 46f71c34c0910f463d69cd6fa75de76ec3edffa07fcf2395c25792c4d2f8eff8 |
| config_toml | config/training/tuned_resnet34_ccsn_15ep.toml | 1164 | 30fa8bb5bcf0e374845a14368d5f060fc45b639daf57248ecb33ff5e4c3d59f6 |
| config_toml | config/training/tuned_resnet34_gcd.toml | 918 | a744cd23ccd8c5b5de684613318d65f797806174d6970e695970e456d23c988a |
| config_toml | config/training/tuned_resnet50.toml | 781 | a615b2efec32cf61c2e7a43d64709d30d2c8ca73a59dae931ef9283c47a57f05 |
| config_toml | config/training/tuned_resnet50_ccsn.toml | 1208 | 890e9d8d1996613438fdd12d98a1374866fee1766103399b3c87aab39665f85f |
| config_toml | config/training/tuned_resnet50_ccsn_15ep.toml | 1164 | d4a01a8416c37edcd9e5572442b5216be3235a05c9b718dcf8c3d6c04feeb8f4 |
| config_toml | config/training/tuned_resnet50_gcd.toml | 918 | d6b8fa3779bc6fd0936f171883b8bae02defb3bbb218b2bab9d1e73b5cd03f4e |

## Inventory Completeness
| Arch | Arm | Seed | Complete | Recorded seed/epochs/reuse | Mismatches |
| --- | --- | --- | --- | --- | --- |
| resnet18 | ccsn15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet18 | ccsn15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet18 | ccsn15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet18 | ccsn90 | 42 | yes | loss:42/90/reuse=False; f1:42/90/reuse=False | none |
| resnet18 | ccsn90 | 43 | yes | loss:43/90/reuse=False; f1:43/90/reuse=False | none |
| resnet18 | ccsn90 | 44 | yes | loss:44/90/reuse=False; f1:44/90/reuse=False | none |
| resnet18 | gcd15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet18 | gcd15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet18 | gcd15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet18 | joint15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet18 | joint15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet18 | joint15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet34 | ccsn15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet34 | ccsn15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet34 | ccsn15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet34 | ccsn90 | 42 | yes | loss:42/90/reuse=False; f1:42/90/reuse=False | none |
| resnet34 | ccsn90 | 43 | yes | loss:43/90/reuse=False; f1:43/90/reuse=False | none |
| resnet34 | ccsn90 | 44 | yes | loss:44/90/reuse=False; f1:44/90/reuse=False | none |
| resnet34 | gcd15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet34 | gcd15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet34 | gcd15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet34 | joint15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet34 | joint15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet34 | joint15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet50 | ccsn15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet50 | ccsn15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet50 | ccsn15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet50 | ccsn90 | 42 | yes | loss:42/90/reuse=False; f1:42/90/reuse=False | none |
| resnet50 | ccsn90 | 43 | yes | loss:43/90/reuse=False; f1:43/90/reuse=False | none |
| resnet50 | ccsn90 | 44 | yes | loss:44/90/reuse=False; f1:44/90/reuse=False | none |
| resnet50 | gcd15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet50 | gcd15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet50 | gcd15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
| resnet50 | joint15 | 42 | yes | loss:42/15/reuse=False; f1:42/15/reuse=False | none |
| resnet50 | joint15 | 43 | yes | loss:43/15/reuse=False; f1:43/15/reuse=False | none |
| resnet50 | joint15 | 44 | yes | loss:44/15/reuse=False; f1:44/15/reuse=False | none |
Recorded provenance gaps: device/git/start/end fields are absent from the per-run `.meta.json`/summary metadata; filesystem mtimes are captured in `evidence_full36.json` as timing proxies.

## Metric Internal Consistency
Recomputed overall accuracy from rounded per-class recalls/support, balanced accuracy from rounded recalls, and macro-F1 from rounded per-class F1. Exceptions within 0.02 pp tolerance: 0.

## Three-Seed Means
### resnet18
| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn15 | ccsn | 58.90 +/- 0.96 | 54.93 +/- 0.47 | 55.28 +/- 1.02 |
| loss | ccsn15 | gcd | 39.14 +/- 4.21 | 46.85 +/- 3.55 | 44.03 +/- 4.13 |
| loss | ccsn90 | ccsn | 61.18 +/- 0.54 | 57.00 +/- 0.25 | 57.51 +/- 0.43 |
| loss | ccsn90 | gcd | 40.05 +/- 2.11 | 47.21 +/- 0.65 | 43.06 +/- 0.78 |
| loss | gcd15 | gcd | 89.88 +/- 0.20 | 92.10 +/- 0.29 | 91.74 +/- 0.13 |
| loss | gcd15 | ccsn | 34.76 +/- 1.42 | 42.24 +/- 1.36 | 33.60 +/- 1.80 |
| loss | joint15 | ccsn | 60.97 +/- 0.54 | 57.73 +/- 0.31 | 58.02 +/- 0.03 |
| loss | joint15 | gcd | 87.22 +/- 1.77 | 88.35 +/- 1.87 | 88.85 +/- 1.81 |
| loss | joint15 | joint | 83.53 +/- 1.59 | 83.84 +/- 1.48 | 84.22 +/- 1.61 |
| loss | joint15 | source_balanced | 74.09 +/- 1.14 | 73.04 +/- 0.80 | 73.44 +/- 0.89 |

| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn15 | ccsn | 59.62 +/- 0.37 | 56.24 +/- 0.88 | 56.50 +/- 0.57 |
| f1 | ccsn15 | gcd | 38.82 +/- 2.64 | 46.44 +/- 2.10 | 44.10 +/- 2.19 |
| f1 | ccsn90 | ccsn | 60.82 +/- 0.86 | 55.84 +/- 1.65 | 56.21 +/- 1.47 |
| f1 | ccsn90 | gcd | 37.61 +/- 1.10 | 45.75 +/- 0.72 | 41.15 +/- 1.37 |
| f1 | gcd15 | gcd | 89.68 +/- 0.41 | 91.68 +/- 0.66 | 91.53 +/- 0.31 |
| f1 | gcd15 | ccsn | 34.90 +/- 1.55 | 42.37 +/- 1.51 | 33.72 +/- 1.92 |
| f1 | joint15 | ccsn | 62.32 +/- 1.42 | 59.68 +/- 1.73 | 59.81 +/- 1.36 |
| f1 | joint15 | gcd | 88.76 +/- 0.30 | 90.29 +/- 0.25 | 90.47 +/- 0.23 |
| f1 | joint15 | joint | 85.05 +/- 0.11 | 85.66 +/- 0.16 | 85.78 +/- 0.12 |
| f1 | joint15 | source_balanced | 75.54 +/- 0.58 | 74.99 +/- 0.75 | 75.14 +/- 0.59 |

### resnet34
| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn15 | ccsn | 58.05 +/- 1.78 | 55.74 +/- 0.65 | 55.35 +/- 0.88 |
| loss | ccsn15 | gcd | 40.67 +/- 0.99 | 49.74 +/- 2.52 | 45.46 +/- 1.91 |
| loss | ccsn90 | ccsn | 57.48 +/- 1.62 | 55.12 +/- 0.70 | 54.79 +/- 0.50 |
| loss | ccsn90 | gcd | 38.87 +/- 1.96 | 48.05 +/- 2.57 | 43.63 +/- 2.04 |
| loss | gcd15 | gcd | 89.33 +/- 0.47 | 91.08 +/- 0.23 | 90.97 +/- 0.29 |
| loss | gcd15 | ccsn | 36.61 +/- 1.17 | 44.47 +/- 1.29 | 35.77 +/- 1.34 |
| loss | joint15 | ccsn | 59.90 +/- 2.48 | 58.12 +/- 1.29 | 57.88 +/- 2.14 |
| loss | joint15 | gcd | 87.59 +/- 2.11 | 88.70 +/- 3.12 | 88.89 +/- 2.94 |
| loss | joint15 | joint | 83.69 +/- 2.12 | 84.23 +/- 2.56 | 84.20 +/- 2.79 |
| loss | joint15 | source_balanced | 73.74 +/- 2.20 | 73.41 +/- 2.15 | 73.39 +/- 2.48 |

| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn15 | ccsn | 57.91 +/- 1.06 | 55.86 +/- 0.72 | 55.53 +/- 1.10 |
| f1 | ccsn15 | gcd | 37.80 +/- 4.06 | 46.79 +/- 4.42 | 42.06 +/- 4.37 |
| f1 | ccsn90 | ccsn | 58.26 +/- 1.71 | 53.56 +/- 0.90 | 53.46 +/- 0.62 |
| f1 | ccsn90 | gcd | 41.03 +/- 2.79 | 47.99 +/- 0.45 | 43.89 +/- 2.07 |
| f1 | gcd15 | gcd | 89.19 +/- 0.53 | 91.16 +/- 0.75 | 90.95 +/- 0.69 |
| f1 | gcd15 | ccsn | 36.25 +/- 1.01 | 44.92 +/- 1.70 | 35.19 +/- 0.92 |
| f1 | joint15 | ccsn | 60.97 +/- 0.65 | 58.31 +/- 0.91 | 58.59 +/- 0.67 |
| f1 | joint15 | gcd | 88.89 +/- 0.24 | 90.49 +/- 0.34 | 90.67 +/- 0.26 |
| f1 | joint15 | joint | 84.96 +/- 0.13 | 85.61 +/- 0.22 | 85.84 +/- 0.15 |
| f1 | joint15 | source_balanced | 74.93 +/- 0.21 | 74.40 +/- 0.43 | 74.63 +/- 0.34 |

### resnet50
| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn15 | ccsn | 59.12 +/- 1.37 | 57.70 +/- 1.77 | 57.24 +/- 1.43 |
| loss | ccsn15 | gcd | 41.13 +/- 3.18 | 48.68 +/- 0.90 | 43.13 +/- 0.52 |
| loss | ccsn90 | ccsn | 58.55 +/- 3.03 | 54.81 +/- 2.70 | 54.77 +/- 2.48 |
| loss | ccsn90 | gcd | 38.17 +/- 4.67 | 46.59 +/- 3.71 | 41.66 +/- 2.18 |
| loss | gcd15 | gcd | 89.49 +/- 0.40 | 91.45 +/- 0.38 | 91.37 +/- 0.13 |
| loss | gcd15 | ccsn | 35.47 +/- 1.86 | 43.76 +/- 3.15 | 34.63 +/- 1.82 |
| loss | joint15 | ccsn | 61.54 +/- 2.94 | 59.03 +/- 1.01 | 58.89 +/- 1.72 |
| loss | joint15 | gcd | 86.90 +/- 1.04 | 88.15 +/- 1.37 | 88.72 +/- 1.11 |
| loss | joint15 | joint | 83.33 +/- 1.29 | 83.80 +/- 1.31 | 84.26 +/- 1.31 |
| loss | joint15 | source_balanced | 74.22 +/- 1.97 | 73.59 +/- 1.17 | 73.80 +/- 1.41 |

| Criterion | Arm | Target | Acc | Bal Acc | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn15 | ccsn | 61.04 +/- 1.01 | 59.30 +/- 1.15 | 59.10 +/- 1.31 |
| f1 | ccsn15 | gcd | 41.00 +/- 2.15 | 46.61 +/- 2.67 | 40.64 +/- 2.72 |
| f1 | ccsn90 | ccsn | 60.47 +/- 2.89 | 57.88 +/- 1.30 | 57.73 +/- 2.06 |
| f1 | ccsn90 | gcd | 42.34 +/- 1.40 | 47.82 +/- 3.03 | 43.07 +/- 1.82 |
| f1 | gcd15 | gcd | 89.73 +/- 0.68 | 91.71 +/- 0.67 | 91.61 +/- 0.48 |
| f1 | gcd15 | ccsn | 35.54 +/- 1.94 | 43.50 +/- 2.87 | 34.47 +/- 1.66 |
| f1 | joint15 | ccsn | 61.90 +/- 1.63 | 57.43 +/- 1.03 | 57.59 +/- 1.06 |
| f1 | joint15 | gcd | 89.66 +/- 0.28 | 91.35 +/- 0.26 | 91.42 +/- 0.32 |
| f1 | joint15 | joint | 85.76 +/- 0.47 | 86.17 +/- 0.36 | 86.52 +/- 0.47 |
| f1 | joint15 | source_balanced | 75.78 +/- 0.96 | 74.39 +/- 0.58 | 74.51 +/- 0.58 |

## Paired Contrasts
### resnet18
| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| loss | Budget_CCSN90_minus_CCSN15 | 2.28 +/- 1.25 (not detectable) | 2.08 +/- 0.67 (detectable) | 2.23 +/- 1.46 (not detectable) |
| loss | Joint_JointOnCCSN_minus_CCSN90 | -0.21 +/- 1.07 (not detectable) | 0.73 +/- 0.44 (not detectable) | 0.51 +/- 0.42 (not detectable) |
| loss | OldClaim_JointOnCCSN_minus_CCSN15 | 2.07 +/- 0.81 (detectable) | 2.81 +/- 0.71 (detectable) | 2.74 +/- 1.03 (detectable) |
| loss | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 4.39 +/- 4.97 (not detectable) | 4.60 +/- 4.34 (not detectable) | 10.43 +/- 5.12 (detectable) |
| loss | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -4.39 +/- 4.97 (not detectable) | -4.60 +/- 4.34 (not detectable) | -10.43 +/- 5.12 (detectable) |

| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| f1 | Budget_CCSN90_minus_CCSN15 | 1.21 +/- 0.62 (not detectable) | -0.40 +/- 1.73 (not detectable) | -0.30 +/- 1.03 (not detectable) |
| f1 | Joint_JointOnCCSN_minus_CCSN90 | 1.50 +/- 0.85 (not detectable) | 3.84 +/- 0.42 (detectable) | 3.61 +/- 0.31 (detectable) |
| f1 | OldClaim_JointOnCCSN_minus_CCSN15 | 2.70 +/- 1.06 (detectable) | 3.44 +/- 1.59 (detectable) | 3.31 +/- 0.85 (detectable) |
| f1 | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 3.92 +/- 4.09 (not detectable) | 4.07 +/- 3.46 (not detectable) | 10.38 +/- 3.82 (detectable) |
| f1 | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -3.92 +/- 4.09 (not detectable) | -4.07 +/- 3.46 (not detectable) | -10.38 +/- 3.82 (detectable) |

### resnet34
| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| loss | Budget_CCSN90_minus_CCSN15 | -0.57 +/- 0.32 (not detectable) | -0.61 +/- 0.54 (not detectable) | -0.56 +/- 0.42 (not detectable) |
| loss | Joint_JointOnCCSN_minus_CCSN90 | 2.42 +/- 0.99 (detectable) | 3.00 +/- 1.74 (not detectable) | 3.09 +/- 1.65 (not detectable) |
| loss | OldClaim_JointOnCCSN_minus_CCSN15 | 1.85 +/- 1.10 (not detectable) | 2.38 +/- 1.26 (not detectable) | 2.53 +/- 1.26 (detectable) |
| loss | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 4.06 +/- 2.08 (not detectable) | 5.28 +/- 3.03 (not detectable) | 9.69 +/- 2.91 (detectable) |
| loss | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -4.06 +/- 2.08 (not detectable) | -5.28 +/- 3.03 (not detectable) | -9.69 +/- 2.91 (detectable) |

| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| f1 | Budget_CCSN90_minus_CCSN15 | 0.35 +/- 1.43 (not detectable) | -2.29 +/- 0.53 (detectable) | -2.07 +/- 0.63 (detectable) |
| f1 | Joint_JointOnCCSN_minus_CCSN90 | 2.71 +/- 1.57 (not detectable) | 4.75 +/- 1.64 (detectable) | 5.13 +/- 0.88 (detectable) |
| f1 | OldClaim_JointOnCCSN_minus_CCSN15 | 3.06 +/- 0.44 (detectable) | 2.46 +/- 1.20 (detectable) | 3.06 +/- 0.91 (detectable) |
| f1 | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 1.54 +/- 3.26 (not detectable) | 1.86 +/- 2.73 (not detectable) | 6.87 +/- 3.77 (not detectable) |
| f1 | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -1.54 +/- 3.26 (not detectable) | -1.86 +/- 2.73 (not detectable) | -6.87 +/- 3.77 (not detectable) |

### resnet50
| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| loss | Budget_CCSN90_minus_CCSN15 | -0.57 +/- 3.51 (not detectable) | -2.89 +/- 1.81 (not detectable) | -2.47 +/- 1.77 (not detectable) |
| loss | Joint_JointOnCCSN_minus_CCSN90 | 2.99 +/- 4.17 (not detectable) | 4.22 +/- 2.82 (not detectable) | 4.12 +/- 2.36 (not detectable) |
| loss | OldClaim_JointOnCCSN_minus_CCSN15 | 2.42 +/- 1.61 (not detectable) | 1.33 +/- 1.27 (not detectable) | 1.64 +/- 0.64 (detectable) |
| loss | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 5.66 +/- 1.46 (detectable) | 4.92 +/- 2.50 (not detectable) | 8.50 +/- 1.47 (detectable) |
| loss | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -5.66 +/- 1.46 (detectable) | -4.92 +/- 2.50 (not detectable) | -8.50 +/- 1.47 (detectable) |

| Criterion | Contrast | Acc diff | Bal Acc diff | Macro-F1 diff |
| --- | --- | --- | --- | --- |
| f1 | Budget_CCSN90_minus_CCSN15 | -0.57 +/- 2.30 (not detectable) | -1.43 +/- 1.69 (not detectable) | -1.38 +/- 1.58 (not detectable) |
| f1 | Joint_JointOnCCSN_minus_CCSN90 | 1.43 +/- 4.52 (not detectable) | -0.45 +/- 1.61 (not detectable) | -0.14 +/- 2.32 (not detectable) |
| f1 | OldClaim_JointOnCCSN_minus_CCSN15 | 0.86 +/- 2.38 (not detectable) | -1.88 +/- 2.17 (not detectable) | -1.51 +/- 2.23 (not detectable) |
| f1 | Transfer_CCSN15onGCD_minus_GCD15onCCSN | 5.46 +/- 1.91 (detectable) | 3.10 +/- 2.23 (not detectable) | 6.17 +/- 2.94 (detectable) |
| f1 | Transfer_GCD15onCCSN_minus_CCSN15onGCD | -5.46 +/- 1.91 (detectable) | -3.10 +/- 2.23 (not detectable) | -6.17 +/- 2.94 (detectable) |

## Transfer Per-Class Recall
### resnet18
| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| loss | cumulus | 40.87 +/- 7.33 | 68.52 +/- 3.20 | -27.65 +/- 9.99 (detectable) |
| loss | altocumulus | 65.88 +/- 4.43 | 60.48 +/- 1.20 | 5.40 +/- 3.27 (not detectable) |
| loss | cirrus | 55.21 +/- 2.81 | 53.73 +/- 2.96 | 1.48 +/- 4.21 (not detectable) |
| loss | stratocumulus | 57.01 +/- 18.88 | 9.59 +/- 2.06 | 47.42 +/- 19.53 (detectable) |
| loss | cumulonimbus | 15.26 +/- 1.46 | 18.91 +/- 3.38 | -3.65 +/- 1.99 (not detectable) |

| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| f1 | cumulus | 39.56 +/- 7.87 | 68.52 +/- 3.20 | -28.96 +/- 10.74 (detectable) |
| f1 | altocumulus | 64.52 +/- 2.04 | 58.76 +/- 1.78 | 5.76 +/- 3.65 (not detectable) |
| f1 | cirrus | 57.74 +/- 5.33 | 55.29 +/- 1.18 | 2.45 +/- 4.18 (not detectable) |
| f1 | stratocumulus | 54.44 +/- 16.15 | 10.04 +/- 2.20 | 44.40 +/- 18.34 (detectable) |
| f1 | cumulonimbus | 15.93 +/- 3.78 | 19.23 +/- 3.85 | -3.30 +/- 5.30 (not detectable) |

### resnet34
| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| loss | cumulus | 46.99 +/- 11.32 | 75.00 +/- 2.78 | -28.01 +/- 8.54 (detectable) |
| loss | altocumulus | 76.95 +/- 3.23 | 59.11 +/- 4.65 | 17.84 +/- 7.63 (detectable) |
| loss | cirrus | 57.13 +/- 8.14 | 53.33 +/- 5.80 | 3.80 +/- 10.30 (not detectable) |
| loss | stratocumulus | 47.99 +/- 11.71 | 12.78 +/- 1.05 | 35.20 +/- 10.85 (detectable) |
| loss | cumulonimbus | 19.66 +/- 3.73 | 22.12 +/- 2.55 | -2.46 +/- 5.22 (not detectable) |

| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| f1 | cumulus | 43.50 +/- 6.41 | 80.55 +/- 7.35 | -37.06 +/- 1.14 (detectable) |
| f1 | altocumulus | 75.93 +/- 3.11 | 59.79 +/- 4.12 | 16.14 +/- 3.14 (detectable) |
| f1 | cirrus | 62.12 +/- 9.71 | 52.94 +/- 3.11 | 9.18 +/- 6.90 (not detectable) |
| f1 | stratocumulus | 27.24 +/- 13.60 | 12.10 +/- 0.79 | 15.15 +/- 13.51 (not detectable) |
| f1 | cumulonimbus | 25.15 +/- 4.17 | 19.23 +/- 2.54 | 5.92 +/- 6.52 (not detectable) |

### resnet50
| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| loss | cumulus | 32.46 +/- 19.34 | 78.70 +/- 13.13 | -46.24 +/- 31.86 (not detectable) |
| loss | altocumulus | 67.91 +/- 4.94 | 57.73 +/- 3.72 | 10.17 +/- 6.70 (not detectable) |
| loss | cirrus | 65.79 +/- 10.54 | 49.41 +/- 4.71 | 16.38 +/- 15.19 (not detectable) |
| loss | stratocumulus | 62.09 +/- 22.96 | 12.78 +/- 3.38 | 49.30 +/- 19.63 (detectable) |
| loss | cumulonimbus | 15.18 +/- 3.23 | 20.19 +/- 1.93 | -5.02 +/- 1.77 (detectable) |

| Criterion | Class | CCSN15 on GCD | GCD15 on CCSN | CCSN-GCD minus GCD-CCSN |
| --- | --- | --- | --- | --- |
| f1 | cumulus | 20.00 +/- 1.83 | 76.85 +/- 11.22 | -56.85 +/- 12.70 (detectable) |
| f1 | altocumulus | 72.09 +/- 7.16 | 61.51 +/- 4.65 | 10.58 +/- 3.88 (detectable) |
| f1 | cirrus | 52.23 +/- 21.05 | 46.28 +/- 2.71 | 5.96 +/- 21.69 (not detectable) |
| f1 | stratocumulus | 75.82 +/- 10.83 | 13.01 +/- 3.62 | 62.81 +/- 9.00 (detectable) |
| f1 | cumulonimbus | 12.89 +/- 5.00 | 19.87 +/- 2.42 | -6.98 +/- 3.42 (detectable) |

## Cross-Architecture Joint Arm
| Criterion | Target | Metric | ResNet-18 | ResNet-34 | ResNet-50 |
| --- | --- | --- | --- | --- | --- |
| loss | ccsn | overall_accuracy | 60.97 +/- 0.54 | 59.90 +/- 2.48 | 61.54 +/- 2.94 |
| loss | ccsn | balanced_accuracy | 57.73 +/- 0.31 | 58.12 +/- 1.29 | 59.03 +/- 1.01 |
| loss | ccsn | macro_f1 | 58.02 +/- 0.03 | 57.88 +/- 2.14 | 58.89 +/- 1.72 |
| Pair | Mean diff | Pooled seed std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| overall_accuracy_resnet18_vs_resnet34 | 1.07 | 1.79 | False |
| overall_accuracy_resnet18_vs_resnet50 | -0.57 | 2.11 | False |
| overall_accuracy_resnet34_vs_resnet50 | -1.64 | 2.72 | False |
| balanced_accuracy_resnet18_vs_resnet34 | -0.39 | 0.94 | False |
| balanced_accuracy_resnet18_vs_resnet50 | -1.30 | 0.75 | False |
| balanced_accuracy_resnet34_vs_resnet50 | -0.91 | 1.16 | False |
| macro_f1_resnet18_vs_resnet34 | 0.14 | 1.51 | False |
| macro_f1_resnet18_vs_resnet50 | -0.86 | 1.22 | False |
| macro_f1_resnet34_vs_resnet50 | -1.01 | 1.94 | False |

| Criterion | Target | Metric | ResNet-18 | ResNet-34 | ResNet-50 |
| --- | --- | --- | --- | --- | --- |
| loss | source_balanced | overall_accuracy | 74.09 +/- 1.14 | 73.74 +/- 2.20 | 74.22 +/- 1.97 |
| loss | source_balanced | balanced_accuracy | 73.04 +/- 0.80 | 73.41 +/- 2.15 | 73.59 +/- 1.17 |
| loss | source_balanced | macro_f1 | 73.44 +/- 0.89 | 73.39 +/- 2.48 | 73.80 +/- 1.41 |
| Pair | Mean diff | Pooled seed std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| overall_accuracy_resnet18_vs_resnet34 | 0.35 | 1.75 | False |
| overall_accuracy_resnet18_vs_resnet50 | -0.12 | 1.61 | False |
| overall_accuracy_resnet34_vs_resnet50 | -0.47 | 2.09 | False |
| balanced_accuracy_resnet18_vs_resnet34 | -0.37 | 1.62 | False |
| balanced_accuracy_resnet18_vs_resnet50 | -0.55 | 1.00 | False |
| balanced_accuracy_resnet34_vs_resnet50 | -0.18 | 1.73 | False |
| macro_f1_resnet18_vs_resnet34 | 0.05 | 1.86 | False |
| macro_f1_resnet18_vs_resnet50 | -0.37 | 1.18 | False |
| macro_f1_resnet34_vs_resnet50 | -0.41 | 2.02 | False |

| Criterion | Target | Metric | ResNet-18 | ResNet-34 | ResNet-50 |
| --- | --- | --- | --- | --- | --- |
| f1 | ccsn | overall_accuracy | 62.32 +/- 1.42 | 60.97 +/- 0.65 | 61.90 +/- 1.63 |
| f1 | ccsn | balanced_accuracy | 59.68 +/- 1.73 | 58.31 +/- 0.91 | 57.43 +/- 1.03 |
| f1 | ccsn | macro_f1 | 59.81 +/- 1.36 | 58.59 +/- 0.67 | 57.59 +/- 1.06 |
| Pair | Mean diff | Pooled seed std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| overall_accuracy_resnet18_vs_resnet34 | 1.35 | 1.11 | False |
| overall_accuracy_resnet18_vs_resnet50 | 0.42 | 1.53 | False |
| overall_accuracy_resnet34_vs_resnet50 | -0.93 | 1.24 | False |
| balanced_accuracy_resnet18_vs_resnet34 | 1.37 | 1.38 | False |
| balanced_accuracy_resnet18_vs_resnet50 | 2.26 | 1.42 | False |
| balanced_accuracy_resnet34_vs_resnet50 | 0.89 | 0.97 | False |
| macro_f1_resnet18_vs_resnet34 | 1.22 | 1.07 | False |
| macro_f1_resnet18_vs_resnet50 | 2.22 | 1.22 | False |
| macro_f1_resnet34_vs_resnet50 | 1.00 | 0.89 | False |

| Criterion | Target | Metric | ResNet-18 | ResNet-34 | ResNet-50 |
| --- | --- | --- | --- | --- | --- |
| f1 | source_balanced | overall_accuracy | 75.54 +/- 0.58 | 74.93 +/- 0.21 | 75.78 +/- 0.96 |
| f1 | source_balanced | balanced_accuracy | 74.99 +/- 0.75 | 74.40 +/- 0.43 | 74.39 +/- 0.58 |
| f1 | source_balanced | macro_f1 | 75.14 +/- 0.59 | 74.63 +/- 0.34 | 74.51 +/- 0.58 |
| Pair | Mean diff | Pooled seed std | Exceeds 2x pooled std |
| --- | --- | --- | --- |
| overall_accuracy_resnet18_vs_resnet34 | 0.61 | 0.44 | False |
| overall_accuracy_resnet18_vs_resnet50 | -0.24 | 0.79 | False |
| overall_accuracy_resnet34_vs_resnet50 | -0.85 | 0.69 | False |
| balanced_accuracy_resnet18_vs_resnet34 | 0.58 | 0.61 | False |
| balanced_accuracy_resnet18_vs_resnet50 | 0.60 | 0.67 | False |
| balanced_accuracy_resnet34_vs_resnet50 | 0.02 | 0.51 | False |
| macro_f1_resnet18_vs_resnet34 | 0.51 | 0.48 | False |
| macro_f1_resnet18_vs_resnet50 | 0.63 | 0.58 | False |
| macro_f1_resnet34_vs_resnet50 | 0.13 | 0.48 | False |

Historical 0.08 pp ResNet-18 vs ResNet-50 CCSN margin: the full min-loss joint CCSN mean difference is -0.57 pp with pooled seed std 2.11; it does not exceed 2x pooled seed std, so it is not distinguishable from noise here.

## Config Verification
| Arch | origin/main approved diff? | CCSN90 ep | CCSN15 ep | CCSN non-epoch diffs | Joint winner | CCSN winner | GCD winner | joint-ref hparam diffs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | no | 90 | 15 | none | True | True | True | none |
| resnet34 | no | 90 | 15 | none | True | True | True | "no_joint_reference_config" |
| resnet50 | no | 90 | 15 | none | True | True | True | "no_joint_reference_config" |

## Reproduction Run
Attempted: `True`. Command: `uv run python src/run_harmonized.py --experiment ccsn --model resnet34 --seed 43 --config config/training/tuned_resnet34_ccsn_15ep.toml --output-dir artifacts/audit_2026-09-07_independent/repro_full36_r34_ccsn15_seed43`.
Completed: `True`; log `artifacts/audit_2026-09-07_independent/reproduction_full36_r34_ccsn15_seed43.log`.
CCSN holdout accuracy reproduced at 57.05% vs recorded seed-43 56.62%; diff 0.43 pp, within 2x seed std = `True`.

## Master Results Table Cross-Check
Checked 102 numeric mean/std cells in Antigravity's master results table against summary-derived aggregates. Discrepancies: 0.

## E1/B1 Status
No machine-readable per-image predictions or confusion matrices were found (`count=0`). PNG confusion plots exist only as figures (`count=82`) and are not machine-readable matrices. Test tensor preload remains at `src/run_harmonized.py:679-707`; evaluation DataLoader construction is at `src/run_harmonized.py:452-457`.

## Final Judgment
The complete 36-run matrix supports CCSN in-domain performance staying in a narrow 57.48-61.54% band under min-loss across CCSN15/CCSN90/Joint-on-CCSN. No architecture shows a robust min-loss joint gain over budget-matched CCSN90 by the requested overall-accuracy screen pattern as a family: ResNet-18 and ResNet-50 are not detectable, while ResNet-34 alone is. Cross-source transfer asymmetry is real but metric/class dependent; the strongest stable signal is macro-F1 and specific class recall directions rather than every top-line accuracy contrast. Architecture differences on the joint arm remain within seed noise for the historical ResNet-18 vs ResNet-50 CCSN question.
