Bilkul, ye poora kaam ek checklist me hai. Har step ke saath "kab ho gaya" bhi likha hai. Kuch likhna nahi hai, jo scripts mil chuki hain unhe chalana hai ya agent se karwana hai.

## Phase 0: 
☐ `PREREG.md` banao. Usme P-S1 (paired MAD ≤ 4), P-S2 (face_out ≤ 0.60), P-S3 (face_only ≥ 0.85), `paired_stats` ke tolerances, aur ek analysis log shuru karo. Natije dekhne se pehle ye likhna zaroori hai.
☐ Teacher se nayi deadline, format/page limit aur template poochho.

## Phase A:
☐ `paired_stats.py` ko `evaluation/` me copy karo.
☐ `linear_probe_beforeafter.py` ki duplicate copy `scripts/` se hata do. Sirf `evaluation/` wali rakho.
☐ Loss ki polarity check karo (`training/losses.py`, `trainer.py`). Kaunsi class "positive" hai, aur kya `pos_weight` ke saath `balanced_sampling` bhi lag raha hai. Jab tak verify na ho, compendium se calibration-mechanism wala hissa hata do.
☐ Git: private remote banao, DF40 ke bade JSONs untrack karo, LICENSE jodo (MIT ya Apache), 33 dirty files commit karo, tag lagao.
☐ Absolute paths hatao (`MASTER_PLAN.md`, compendium, 2 tests). `scratch/` aur extra `.md` files archive karo.
☐ Purani `RESULTS_LEDGER.json` archive karo. Ab se har number script ke ledger se aayega.
☐ Compendium/MASTER_PLAN se architecture ki details hatao ya theek karo. FaceShifter ka forgery_type "composited" karo.
☐ `pip freeze` se environment file banao.
☐ Main DF40-pairing linear-probe run final script se dobara chalao (ledger consistent ho).
☐ `models/dino_branch.py` ka secret-like hit dekh lo (shayad false positive).

**Done jab:** collector me "STALE" ledger koi na bache, aur git me remote ho.

## Phase B: 
☐ `extract_shared_transform_pairs.py` chalao (600 pairs, 4 frames, official split folder do). 20 random crops dekho: real aur fake ka face same jagah aur same scale par hona chahiye.
☐ Isi data par chalao: paired_diagnostic, paired_stats, region_probe, aur linear probe (full, face_out, face_only).
☐ Natije ke hisaab se:
- **MAD ≤ 4 aur face_out ≤ 0.60:** alignment hi kaaran tha. Shared-transform pairing tumhari clean reference ban gayi.
- **face_out ~0.74 hi raha:** alignment kaaran nahi. Phir ye dekho: compression generation, face ke paas ka spill (mask ka radius 0.7/0.85/1.0 karke), aur kaunsa method sabse zyada contribute kar raha hai.
☐ Official FF++ split (720/140/140) use karo. Agent se kaho ki linear probe me split option jode.
☐ Har linear-probe table 5 seeds se chalao, mean ± sd aur CI likho.
☐ LOGO dono pairings par aise karo ki held-out generators ek jaise hon, aur jo alag hon wo likh do.
☐ Size aur identity controls ko 5 seeds se dobara chalao.

## Phase C: 
☐ Doosra aur teesra backbone: CLIP ViT-B/16 aur ResNet-50 (frozen), wahi sab probes.
☐ Ek chhota fine-tuned CNN (EfficientNet-B0, 8GB me) dono pairings par, aur cross-pairing transfer dekho.
☐ Doosra domain: Celeb-DF native pairing aur DF40-cdf wali pairing. Phase B ke saare steps dohrao.
☐ Optional: DeepfakeBench ke 1–2 pretrained weights agar mil sakein, to confounded aur shared-transform pairing par evaluate karo.
☐ Dataset-identity baseline ko ek table me daalo (label sirf dataset ID se 0.921 AUROC).

## Phase D: 
☐ Scripts freeze karo (tag), phir saare runs saaf state se ek baar aur karo. Collector me stale ledger zero ho.
☐ Tables sirf ledgers se (`make_latex_tables`): pairing, region controls, LOGO, dataset-ID baseline, second backbone.
☐ Figures: crop montage (shared-transform vs alag-alag align), region/pairing bar chart, LOGO ka plot, audit pipeline ka diagram.
☐ Limitations ki list aur analysis log final karo.

## Phase E:
☐ Structure: Abstract, Intro, Related Work, Audit Protocol, Setup, Results, Analysis, Limitations, Recommendations (benchmark builders ke liye checklist), Conclusion, Appendix (ledger hashes, PREREG, analysis log).
☐ Intro, Discussion aur Recommendations tum likho. Methods, tables aur appendix ka draft main dunga. Phir dono ek-doosre ka review karenge.
☐ Related work me 25–40 references, har ek publisher page se verify karke. Zaroor: Grommelt et al., DeepfakeBench, DF40, FF++, Celeb-DF, Geirhos (shortcut learning), Lapuschkin (Clever Hans).
☐ Har claim ke saath table, aur "in our setting" likhna. "genuine", "proves", "zero-shot" mat likhna.
☐ Adaptive analyses disclose karo, aur AI use me Claude aur Antigravity dono ka naam do.

## Phase F: r
☐ README me reproduce karne ke 5 steps. Sirf scripts, ID lists aur ledgers release karo, images nahi.
☐ FF++, Celeb-DF aur DF40 ke license/forms check karo.
☐ Zenodo DOI, GitHub tag, arXiv (cs.CV; pehli baar endorsement lag sakta hai, teacher se poochho).
☐ CVPR 2027 aur ICCV 2027 workshops ki list aur deadlines jab aayen tab khud dekhna.

## Detector paper (ye sab ke baad)
☐ Training set clean pairings se dobara banao. Har dataset me real:fake ka ratio balanced rakho taaki dataset ID label na bataye.
☐ Loss aur sampling theek karo, training logging lagao.
☐ Branch ablations retrain karke. Video-level evaluation. Abstention ko mushkil data par dobara test karo.

## Mere liye kab bhejna hai
Sirf `HANDOFF.md`, teen jagah: Phase A ke baad, Phase B ke baad (saath me PREREG ke natije), aur Phase C ke baad. Main har baar bataunga ki agla kya.

## Hamesha ke niyam
- Test data dekh kar kuch tune mat karo.
- Run se pehle commit karo, aur koi number haath se mat badlo.
- Is paper me FORGE-Lite ke branches ya abstention ke naye claims mat jodo.
- Native pairing ko "clean" tab hi kaho jab Phase B ke natije aa jayen.