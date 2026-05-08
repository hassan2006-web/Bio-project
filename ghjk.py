import os # مكتبة التعامل مع نظام التشغيل: في بيئة المعلوماتية الحيوية، نستخدمها لإدارة قواعد البيانات الضخمة (Big Data) مثل مجلد "Protein_Database_100" والتحقق من وجود الملفات الهيكلية (.pdb, .ent) قبل معالجتها.
import json # مكتبة معالجة البيانات المهيكلة: تُستخدم لتبادل البيانات بين السيرفر والتطبيق. هنا نستخدمها لتحميل خريطة الطفرات (Mutation Map) التي تربط البروتينات الطافرة بنظيراتها البرية (Wild-type) بشكل مؤتمت.
import time # مكتبة التحكم بالوقت: نستخدمها هنا لعمل (Cache Busting)؛ أي إضافة رقم متغير للروابط لضمان جلب أحدث نسخة من قاعدة بيانات الطفرات من GitHub وعدم الاعتماد على النسخ القديمة المخزنة في الذاكرة.
import requests # مكتبة طلبات الشبكة: هي الجسر الذي يربط تطبيقنا بقواعد البيانات العالمية مثل RCSB PDB. تسمح لنا بجلب بنية البروتين (3D Structure) فوراً بمجرد إدخال الكود المكون من 4 رموز (مثل 1A2B).
import numpy as np # مكتبة الحوسبة العددية: البروتين في الحاسوب هو عبارة عن آلاف النقاط في فراغ ثلاثي الأبعاد (X, Y, Z). Numpy تسمح لنا بإجراء عمليات رياضية "متجهة" (Vectorized) على هذه النقاط بسرعة هائلة تفوق لغة بايثون العادية بآلاف المرات.
import pandas as pd # مكتبة تحليل الجداول: في علم الـ Bioinformatics، نحتاج لترتيب النتائج (مثل قائمة الأحماض المتأثرة بالطفرة) في جداول (DataFrames) لسهولة إحصاء التغيرات وحساب نسبة التشابه (Identity).
from io import StringIO # أداة الإدخال والإخراج النصي: تسمح لنا بمعاملة النصوص الجاهزة (PDB Strings) وكأنها ملفات حقيقية مخزنة على القرص الصلب، مما يسرع عملية المعالجة ويقلل استهلاك الذاكرة.
from scipy.spatial import KDTree # خوارزمية البحث المكاني (Spatial Search): تُستخدم لحل مشكلة "أقرب جار". في البروتينات، نستخدمها لاكتشاف التفاعلات بين الذرات (Interactions)؛ فبدلاً من حساب المسافة بين كل ذرة وجميع الذرات الأخرى (عملية ثقيلة جداً)، يقوم الـ KDTree بالبحث فقط في المحيط المجاور.
from Bio.PDB import PDBParser # محلل Biopython: يحول ملف الـ PDB النصي المعقد إلى كائن هرمي برمجياً. هذا المحلل يفهم قواعد الكيمياء الحيوية، فيعرف أين تنتهي السلسلة وأين يبدأ الحمض الأميني وأين تقع كل ذرة داخله.
from Bio.PDB.SASA import ShrakeRupley # خوارزمية حساب مساحة السطح (SASA): من أهم المقاييس في علم البروتينات. تحسب مقدار تعرض كل حمض أميني للماء. الحمض المدفون (Buried) يكون محمياً في قلب البروتين، بينما المكشوف (Exposed) يكون متفاعلاً مع البيئة المحيطة.
from Bio.Align import PairwiseAligner # أداة محاذاة السلاسل: تستخدم خوارزميات (Dynamic Programming) مثل Needleman-Wunsch لمقارنة تسلسل بروتينين. هي الأساس في معرفة التطور الجيني (Evolutionary relationship) وتحديد مواقع الطفرات بدقة.
import streamlit as st # إطار عمل الواجهة: يحول أكواد البايثون العلمية إلى تطبيق ويب تفاعلي سهل الاستخدام للباحثين والأطباء دون الحاجة لخبرة في البرمجة.
import streamlit.components.v1 as components # أداة التضمين: تسمح لنا بحقن أكواد (JavaScript/HTML) الخارجية داخل Streamlit، وهي ضرورية لعرض النوافذ التفاعلية ثلاثية الأبعاد.
import py3Dmol # مكتبة العرض الجزيئي: تستخدم تقنيات WebGL لعرض البروتين بشكل 3D. تسمح لنا بتمثيل البروتين بأنماط مختلفة (Cartoon لمسارات السلسلة، Sticks للروابط الكيميائية، Spheres للذرات) لإبراز تفاصيل الطفرات.

# ============================================================
# ثوابت المعلوماتية الحيوية (Bioinformatics Constants)
# ============================================================

# قاموس تحويل الأحماض من 3 أحرف إلى حرف واحد.
# علمياً: الرموز الثلاثية (مثل GLY) تُستخدم في الكيمياء الهيكلية، بينما الرموز الأحادية (G) تُستخدم في علم الجينوم والمحاذاة التسلسلية لتسهيل الحسابات الرياضية والمقارنة النصية.
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# تصنيف الأحماض حسب الخواص الفيزيائية والكيميائية (Physicochemical Properties).
# هذه الخصائص هي المحرك الأساسي لعملية "الطي البروتيني" (Protein Folding).
AA_PROPS = {
    'ALA': 'Non-polar', 'GLY': 'Non-polar', 'ILE': 'Non-polar', 'LEU': 'Non-polar', 'MET': 'Non-polar', 'PRO': 'Non-polar', 'VAL': 'Non-polar', # كارهة للماء (Hydrophobic): تشكل "اللب الكاره للماء" في قلب البروتين بعيداً عن الماء المحيط.
    'ASN': 'Polar',     'CYS': 'Polar',     'GLN': 'Polar',     'SER': 'Polar',     'THR': 'Polar', # قطبية (Polar): محبة للماء، تتواجد غالباً على السطح لتتفاعل مع جزيئات الماء أو الروابط الهيدروجينية.
    'ASP': 'Acidic (-)', 'GLU': 'Acidic (-)', # حمضية (شحنة سالبة): تشكل روابط أيونية (Salt Bridges) مع الأحماض القاعدية لتثبيت البنية.
    'ARG': 'Basic (+)',  'HIS': 'Basic (+)',  'LYS': 'Basic (+)', # قاعدية (شحنة موجبة).
    'PHE': 'Aromatic',   'TRP': 'Aromatic',   'TYR': 'Aromatic' # عطرية (Aromatic): ضخمة الحجم وتحتوي على حلقات كربونية، مهمة جداً في استقرار البروتين عبر تفاعلات (Pi-stacking).
}

def analyze_impact(h_res, m_res, h_sasa, m_sasa):
    """
    تحليل التأثير العلمي العميق للطفرة (Structural & Chemical Impact Prediction).
    في علم الأمراض الجزيئي، نبحث عن التغيرات التي تؤدي لخلل وظيفي (Pathogenic Mutations).
    """
    # إذا لم يتغير الحمض، الطفرة تسمى "محافظة" (Conservative) وغالباً لا تؤثر على الوظيفة.
    if h_res == m_res: return "Conservative"
    
    h_type = AA_PROPS.get(h_res, 'Unknown')
    m_type = AA_PROPS.get(m_res, 'Unknown')
    
    impacts = []
    
    # 1. تحليل التغير الكيميائي والشحنة:
    if h_type != m_type:
        # انقلاب الشحنة (Charge Flip): من أخطر أنواع الطفرات، لأنه يكسر الروابط الأيونية التي تربط أجزاء البروتين ببعضها، مما قد يؤدي لانهيار البنية بالكامل.
        if "Acidic" in h_type and "Basic" in m_type or "Basic" in h_type and "Acidic" in m_type:
            impacts.append("Charge Flip (Critical)")
        else:
            # تغير الفئة الكيميائية (مثلاً من كاره للماء إلى قطبي): قد يؤدي لدخول الماء لقلب البروتين وتدمير استقراره.
            impacts.append("Chem-Class Change")
    
    # 2. تحليل التغير الهيكلي عبر الـ SASA (مساحة التعرض):
    try:
        diff = m_sasa - h_sasa
        # إذا كان الفرق كبيراً (> 10 أنجستروم مربع)، فهذا مؤشر على تغير في "ديناميكا البروتين".
        if abs(diff) > 10:
            # Exposed: الحمض أصبح مكشوفاً، ربما بسبب فتح جزء من البروتين.
            # Buried: الحمض أصبح مدفوناً، ربما بسبب انكماش في البنية.
            impacts.append("Exposed" if diff > 0 else "Buried")
    except:
        pass
        
    # إرجاع وصف علمي متكامل للتأثير المكتشف.
    return " | ".join(impacts) if impacts else "Minor Change"

# ============================================================
# العمليات الحيوية المؤتمتة (Automated Biological Pipelines)
# ============================================================

@st.cache_data(ttl=3600) # تخزين الكاش: في الأبحاث العلمية، جلب البيانات وتكرار معالجتها يستهلك وقتاً، الكاش يضمن سرعة الاستجابة عند تكرار نفس التحليل.
def fetch_pdb_data(pdb_id):
    """جلب بيانات البروتين من المستودعات المحلية أو العالمية (Data Retrieval)."""
    if not pdb_id or pdb_id == "NONE":
        return None  
    
    # التحقق من قاعدة بياناتك المحلية الـ 100 بروتين (Local Database lookup).
    local_dir = "Protein_Database_100"
    local_file_path = os.path.join(local_dir, f"pdb{pdb_id.lower()}.ent")
    
    if os.path.exists(local_file_path):
        try:
            with open(local_file_path, "r", encoding="utf-8") as f:
                return f.read() # قراءة الهيكل 3D من الجهاز مباشرة.
        except Exception as error:
            st.warning(f"فشل القراءة المحلية: {error}")

    # جلب البيانات من RCSB PDB (Global Database access).
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.error(f"الكود {pdb_id} غير موجود في السجلات العالمية.")
            return None
    except requests.exceptions.RequestException as error:
        st.error(f"خطأ في الاتصال بالسيرفر: {error}")
        return None

@st.cache_data(ttl=3600)
def process_protein_structure(pdb_string, pdb_id, keep_hetatm=False):
    """تحويل النص الخام إلى "كائن حي برمجياً" (Structure Parsing & Cleaning)."""
    try:
        parser = PDBParser(QUIET=True)
        # إنشاء هيكل هرمي يسمح لنا بالوصول لكل ذرة وحمض أميني بشكل مستقل.
        structure = parser.get_structure(pdb_id, StringIO(pdb_string))
        
        # تنظيف الهيكل (Structure Cleaning): ملفات الـ PDB تحتوي غالباً على جزيئات ماء أو شوائب كيميائية من تجربة الكريستالوكرافي. 
        # نقوم بحذفها (HetAtoms) للتركيز فقط على السلسلة البروتينية الصافية.
        if not keep_hetatm:
            for model in structure:
                for chain in model:
                    # استبعاد كل ما هو ليس حمضاً أمينياً قياسياً.
                    for r_id in [r.get_id() for r in chain if r.get_id()[0] != ' ']:
                        chain.detach_child(r_id)
        
        # حساب الـ SASA فوراً: نستخدم خوارزمية Shrake-Rupley التي تمرر "كرة وهمية" حول البروتين لمحاكاة جزيء الماء، 
        # وذلك لحساب المساحة التي يمكن للماء لمسها من سطح كل حمض أميني.
        try:
            sr = ShrakeRupley()
            sr.compute(structure, level='R') 
        except:
            pass
            
        return structure
    except Exception as error:
        st.error(f"خطأ في معالجة الهيكل: {error}")
        return None

def get_all_chains(structure):
    """استخراج سلاسل الببتيد (Polypeptide Chains). البروتين قد يتكون من سلسلة واحدة أو عدة سلاسل تعمل معاً."""
    try:
        return [chain.id for chain in structure[0]]
    except Exception as ex:
        st.error(f"فشل قراءة السلاسل: {ex}")
        return []

def get_protein_sequence(structure, chain_id):
    """تحويل الهيكل الفراغي 3D إلى "شيفرة نصية" (Sequence Extraction). هي الخطوة الأولى للمقارنة الجينية."""
    sequence = []
    try:
        model = structure[0]
        if chain_id in [c.id for c in model]:
            for residue in model[chain_id]:
                # الأحماض القياسية فقط (Standard Amino Acids).
                if residue.id[0] == ' ':
                    sequence.append({
                        'res_num': residue.id[1], # رقم الحمض (موقع الطفرة).
                        'res_name': residue.get_resname() # اسم الحمض (الهوية الكيميائية).
                    })
    except Exception as ex:
        st.error(f"فشل استخراج التسلسل: {ex}")
    return sequence

def sequence_to_fasta(structure, chain_id, protein_name="protein"):
    """تصدير التسلسل بصيغة FASTA. هي اللغة العالمية التي تفهمها جميع برامج البيولوجيا الجزيئية (مثل BLAST)."""
    seq_data = get_protein_sequence(structure, chain_id)
    if not seq_data: return None
    # التحويل من الأسماء الثلاثية إلى الأحادية (مثلاً ALA -> A).
    one_letter = ''.join([AA_3TO1.get(r['res_name'], 'X') for r in seq_data])
    # تقسيم النص لأسطر (تنسيق FASTA القياسي).
    lines = [one_letter[i:i+80] for i in range(0, len(one_letter), 80)]
    header = f">{protein_name}|Chain_{chain_id}|length={len(one_letter)}\n"
    return header + '\n'.join(lines) + '\n'

@st.cache_data(ttl=3600)
def calculate_all_distances(_structure_key, pdb_string, chain_id, radius=5.0, keep_hetatm=False):
    """
    تحليل شبكة التفاعلات (Residue Interaction Network - RIN).
    في علم البروتينات، لا يعمل الحمض الأميني بمفرده؛ بل عبر التفاعل مع جيرانه.
    """
    structure = process_protein_structure(pdb_string, "prot", keep_hetatm)
    if not structure: return []
    
    try:
        model = structure[0]
        if chain_id not in [c.id for c in model]: return []

        # استخراج كافة الذرات وتحويلها لمصفوفة Numpy للعمليات الرياضية السريعة (Matrix operations).
        all_atoms = list(model.get_atoms())
        all_coords = np.array([a.get_coord() for a in all_atoms], dtype=np.float32)
        
        # بناء KDTree: خوارزمية بحث متقدمة تقسم الفراغ 3D لمناطق، مما يسمح لنا بإيجاد الذرات المتفاعلة في زمن قياسي (O(log n)).
        tree = KDTree(all_coords)
        
        # تخزين بيانات تعريفية لكل ذرة لربط النتائج بالحمض الأميني الصحيح.
        atom_info = []
        for a in all_atoms:
            res = a.get_parent()
            atom_info.append((res.get_parent().id, res.id[1]))
        
        residues = [r for r in model[chain_id] if r.id[0] == ' ']
        results = []

        for target_res in residues:
            target_res_id = target_res.id[1]
            target_coords = np.array([a.get_coord() for a in target_res.get_atoms()], dtype=np.float32)
            
            # البحث عن جميع الذرات التي تقع ضمن "نطاق التأثير الكيميائي" (Radius).
            # علمياً: المسافة 3-5 أنجستروم هي النطاق الذي تحدث فيه معظم الروابط الهيدروجينية وقوى فان دير فالس.
            indices = tree.query_ball_point(target_coords, radius)
            
            nearby_indices = set()
            for idx_list in indices:
                for idx in idx_list:
                    info = atom_info[idx]
                    # استبعاد الذرات التي هي جزء من نفس الحمض الأميني.
                    if not (info[0] == chain_id and info[1] == target_res_id):
                        nearby_indices.add(idx)

            # حساب أقل مسافة فعلية (Minimum Euclidean Distance) بين الحمض المستهدف وأي ذرة محيطة.
            min_dist = "-"
            if nearby_indices:
                nb_coords = all_coords[list(nearby_indices)]
                # طرح مصفوفات الإحداثيات للحصول على فروق المسافات (Vector Subtraction).
                diff = target_coords[:, np.newaxis, :] - nb_coords[np.newaxis, :, :]
                dist_sq = (diff * diff).sum(axis=-1)
                min_dist = round(float(np.sqrt(dist_sq.min())), 2)

            resname = target_res.get_resname()
            # جلب قيمة SASA المحسوبة مسبقاً.
            sasa_val = getattr(target_res, 'sasa', '-')
            if isinstance(sasa_val, (float, int)):
                sasa_val = round(sasa_val, 2)

            # تجميع البيانات في سجل علمي متكامل لكل حمض أميني.
            results.append({
                'chain'          : chain_id,
                'res_num'        : target_res_id,
                'res_name'       : resname,
                'one_letter'     : AA_3TO1.get(resname, 'X'),
                'class'          : AA_PROPS.get(resname, '-'),
                'min_dist'       : min_dist,
                'sasa'           : sasa_val
            })

        return results
    except Exception as error:
        st.error(f"خطأ في التحليل الهيكلي: {error}")
        return []

def calculate_sasa_map(structure, chain_id):
    """إنشاء خريطة SASA (Solvent Accessibility Mapping). تساعد في مقارنة "درجة الانطواء" بين السليم والمصاب."""
    try:
        sasa_map = {}
        for res in structure[0][chain_id]:
            if res.id[0] == ' ':
                sasa_map[res.id[1]] = round(getattr(res, 'sasa', 0), 2)
        return sasa_map
    except Exception:
        return {}

def render_protein_3d(pdb_string, bg_color='#0E1117', style_type='cartoon',
                      show_surface=True, surface_opacity=0.3, mutations=None,
                      mut_color='red', zoom_to_mutations=False,
                      focus_mut=None, keep_hetatm=False):
    """توليد الواجهة الرسومية ثلاثية الأبعاد (Molecular Visualization). التبسيط البصري ضروري لفهم التعقيد الحيوي."""
    view = py3Dmol.view(width="100%", height=450)
    view.addModel(pdb_string, 'pdb')
    view.setBackgroundColor(bg_color)

    # التلوين بنظام الطيف (Spectrum): يلون من الأزرق (البداية N-terminus) للأحمر (النهاية C-terminus).
    # يساعد الباحث في تتبع مسار طي السلسلة الببتيدية.
    style_dict = {style_type: {'color': 'spectrum'}}
    view.setStyle({'model': -1}, style_dict)

    if keep_hetatm:
        # عرض الأدوية (Ligands) بستايل العصي (Sticks) لتمييزها عن البروتين.
        view.addStyle({'hetflag': True}, {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.2}})

    if show_surface:
        # إضافة السطح الجزيئي (Molecular Surface): يوضح الفراغات والجيوب (Pockets) التي قد ترتبط بها الأدوية.
        view.addSurface(py3Dmol.SAS, {'opacity': surface_opacity, 'color': '#FFC107'})

    # إبراز الطفرات (Mutation Highlighting): تلوين الحمض الأميني الطافر بلون صارخ وتكبير حجم ذراته.
    if mutations:
        for mut in mutations:
            view.addStyle(mut, {style_type: {'color': mut_color}})
            view.addStyle(mut, {'stick'  : {'colorscheme': 'yellowCarbon', 'radius': 0.3}})
            view.addStyle(mut, {'sphere' : {'color': mut_color, 'radius': 1.2}})

    # التحكم بالكاميرا (Camera Control): التقريب التلقائي لموقع الطفرة لفحصها عن قرب.
    if focus_mut:
        view.zoomTo(focus_mut)
    elif zoom_to_mutations and mutations:
        view.zoomTo({'or': mutations})
    else:
        view.zoomTo()

    return view._make_html()

def get_alignment(seq1, seq2, mode='global'):
    """المحاذاة التسلسلية (Pairwise Alignment). هي عملية مقارنة "نصية" بين الأحماض لمعرفة التغيرات الجينية."""
    aligner = PairwiseAligner()
    aligner.mode = mode # Global: للمقارنة الكاملة من الطرف للطرف. Local: للبحث عن قطع متشابهة داخل سلاسل مختلفة.
    try:
        # خوارزمية المحاذاة تعطي درجات (Scores) بناءً على التشابه، مع خصم نقاط عند وجود فجوات (Gaps).
        best_aln = aligner.align(seq1, seq2)[0]
        return str(best_aln), best_aln.score, best_aln[0], best_aln[1]
    except Exception as error:
        st.warning(f"فشل المحاذاة: {error}")
        return "", 0, "", ""

# رابط قاعدة بيانات الطفرات المؤتمتة (Automated Mutation Database).
REMOTE_JSON_URL = "https://raw.githubusercontent.com/hassan2006-web/Bio-project/refs/heads/main/mutations.json"

@st.cache_data(ttl=300)
def load_mutation_db():
    """تحميل خريطة الطفرات من GitHub. تسمح للتطبيق بالتعرف على "الزوج السليم" تلقائياً بمجرد إدخال كود المصاب."""
    url = f"{REMOTE_JSON_URL}?t={int(time.time())}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}

# ============================================================
# مكونات واجهة المستخدم التفاعلية (Interactive UI Components)
# ============================================================

def protein_ui_panel(p, keep_hetatm):
    """بناء لوحة التحكم في البروتين (Input & Data Management Panel)."""
    with p["col"]:
        icon = '🟢' if p['prefix'] == 'h' else '🔴'
        st.header(f"{icon} {p['label']}")
        source = st.radio("المصدر:", ["PDB ID", "رفع ملف"], key=f"{p['prefix']}_src", horizontal=True)

        if source == "PDB ID":
            pdb_input = st.text_input("كود PDB (مثال: 1A2B):", key=f"{p['prefix']}_id_in").strip().upper()

            if st.button(f"تحميل وتجهيز {p['label']}", key=f"btn_{p['prefix']}"):
                with st.spinner('جاري الاتصال بقاعدة البيانات...'):
                    # الأتمتة الذكية: إذا أدخل المستخدم كود المصاب (Mutant)، يقوم التطبيق تلقائياً بالبحث عن البروتين السليم المقابل له (Wild-type).
                    if p['prefix'] == 'm':
                        mdb = load_mutation_db()
                        if pdb_input in mdb:
                            h_id = mdb[pdb_input]
                            st.session_state['h_id_in'] = h_id
                            h_data = fetch_pdb_data(h_id)
                            if h_data:
                                st.session_state["h_pdb"] = h_data
                                st.session_state["h_id"]  = h_id

                    data = fetch_pdb_data(pdb_input)
                    if data:
                        st.session_state[f"{p['prefix']}_pdb"] = data
                        st.session_state[f"{p['prefix']}_id"]  = pdb_input
                        # تنظيف حالة الجلسة (Session State Cleanup) لضمان عدم تداخل بيانات بروتينات قديمة.
                        for k in list(st.session_state.keys()):
                            if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_id_in"]:
                                st.session_state.pop(k, None)
                        st.rerun()
                    else:
                        st.error(f"لم يتم العثور على البروتين: {pdb_input}")
        else:
            file = st.file_uploader(f"ارفع ملف {p['label']} بصيغة .pdb:", type=["pdb"], key=f"{p['prefix']}_up")
            if file:
                # قراءة البيانات مباشرة من الملف المرفوع وتحويلها لنص.
                st.session_state[f"{p['prefix']}_pdb"] = file.getvalue().decode("utf-8")
                st.session_state[f"{p['prefix']}_id"]  = file.name
                for k in list(st.session_state.keys()):
                    if k.startswith(p['prefix']) and k not in [f"{p['prefix']}_pdb", f"{p['prefix']}_id", f"{p['prefix']}_src", f"{p['prefix']}_up"]:
                        st.session_state.pop(k, None)

def initialize_session_state():
    """تهيئة ذاكرة التطبيق (State Initialization). تضمن استقرار التطبيق عند إعادة التحميل."""
    defaults = {
        'h_pdb': None, 'm_pdb': None,
        'h_id': '', 'm_id': '',
        'h_results': None, 'm_results': None,
        'h_selected_chain': None, 'm_selected_chain': None,
        'h_id_in': '', 'm_id_in': ''
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def main():
    """نقطة انطلاق التطبيق (Main Orchestrator). تنظم تدفق العمل من الإدخال حتى التحليل النهائي."""
    st.set_page_config(page_title="Bio-Impact Analyzer", page_icon="🧬", layout="wide")
    initialize_session_state()
    st.title("🧬 Bio-Impact Analyzer Professional")
    st.markdown("تحليل هيكلي وكيميائي حيوي متقدم لمقارنة البروتينات السليمة والطافرة.")

    # ── إعدادات التحليل (Analysis Configuration Sidebar) ──
    st.sidebar.header("⚙️ معايير التحليل الهيكلي")
    with st.sidebar.expander("🎨 التخصيص البصري", expanded=True):
        search_radius  = st.slider("🔍 مدى التفاعل الكيميائي (Å)", 3.0, 12.0, 5.0)
        view_style     = st.selectbox("أسلوب التمثيل الجزيئي", ["cartoon", "stick", "sphere"])
        show_surf      = st.checkbox("إظهار السطح المتاح للمذيب", value=False)
        surface_op     = st.slider("شفافية السطح", 0.0, 1.0, 0.3)
    
    with st.sidebar.expander("🧬 خيارات الطفرات والمحاذاة", expanded=False):
        show_mutations = st.checkbox("تظليل مواقع الطفرات", value=True)
        zoom_mutations = st.checkbox("تركيز تلقائي على الطفرات", value=False)
        align_mode     = st.selectbox("خوارزمية المحاذاة", ["global", "local"])
        keep_hetatm    = st.checkbox("💊 الإبقاء على الجزيئات الدوائية", value=False)

    col1, col2 = st.columns(2)
    proteins = [
        {"label": "البروتين المصاب (Mutated)", "prefix": "m", "col": col2, "bg": "#1E0D0D"},
        {"label": "البروتين السليم (Healthy)", "prefix": "h", "col": col1, "bg": "#0D1B1E"}
    ]

    # 1. مرحلة إدخال البيانات (Data Input Phase).
    for p in proteins:
        protein_ui_panel(p, keep_hetatm)

    st.divider()

    # 2. مرحلة المعالجة والعرض (Processing & Visualization Phase).
    v_col1, v_col2 = st.columns(2)
    structures  = {}
    
    for p in proteins:
        current_col = v_col1 if p['prefix'] == 'h' else v_col2
        with current_col:
            pdb_data = st.session_state.get(f"{p['prefix']}_pdb")
            if not pdb_data: continue

            st.subheader(f"بنية {p['label']}")
            # بناء كائن الهيكل (Structure Object Generation).
            struct = process_protein_structure(pdb_data, p['prefix'], keep_hetatm)
            structures[p['prefix']] = struct

            if struct:
                chains = get_all_chains(struct)
                if chains:
                    # اختيار السلسلة: في البروتينات الضخمة، قد نهتم بسلسلة واحدة فقط (مثل بروتين Spike في كورونا).
                    selected_chain = st.selectbox(f"تحديد السلسلة للتحليل - {p['label']}", options=chains, key=f"{p['prefix']}_chain_sel")
                    st.session_state[f"{p['prefix']}_selected_chain"] = selected_chain
                else:
                    st.warning("لم يتم العثور على سلاسل ببتيدية.")

    # 3. مرحلة اكتشاف الطفرات (Mutation Detection Logic).
    highlight_map = {'h': None, 'm': None}
    h_chain = st.session_state.get('h_selected_chain')
    m_chain = st.session_state.get('m_selected_chain')
    
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        h_seq = get_protein_sequence(structures['h'], h_chain)
        m_seq = get_protein_sequence(structures['m'], m_chain)
        if h_seq and m_seq:
            # استخدام تقنية القواميس (Mapping) لمقارنة كل موقع في البروتينين.
            dict_h = {r['res_num']: r['res_name'] for r in h_seq}
            dict_m = {r['res_num']: r['res_name'] for r in m_seq}
            mut_m = []
            mut_h = []
            # دمج أرقام الأحماض من البروتينين لمعرفة الفروقات.
            for res_num in set(dict_h) | set(dict_m):
                if dict_h.get(res_num) != dict_m.get(res_num):
                    if res_num in dict_m: mut_m.append({'resi': str(res_num), 'chain': str(m_chain)})
                    if res_num in dict_h: mut_h.append({'resi': str(res_num), 'chain': str(h_chain)})
            highlight_map['m'] = mut_m if mut_m else None
            highlight_map['h'] = mut_h if mut_h else None

    # 4. مرحلة العرض ثلاثي الأبعاد والتحليل الفردي (Visualization & Single Analysis).
    for p in proteins:
        current_col = v_col1 if p['prefix'] == 'h' else v_col2
        with current_col:
            prefix = p['prefix']
            pdb_data = st.session_state.get(f"{prefix}_pdb")
            struct = structures.get(prefix)
            selected_chain = st.session_state.get(f"{prefix}_selected_chain")

            if not pdb_data or not struct or not selected_chain: continue

            highlight = highlight_map[prefix]
            focus_mut = None
            if highlight:
                mut_opts = ["عرض كامل الهيكل"] + [f"موقع {m['resi']} (سلسلة {m['chain']})" for m in highlight]
                sel_mut  = st.selectbox(f"🔍 فحص طفرة محددة - {p['label']}", mut_opts, key=f"focus_{prefix}")
                if sel_mut != "عرض كامل الهيكل":
                    parts     = sel_mut.split(" ")
                    focus_mut = {'resi': parts[1], 'chain': parts[3].replace(")", "")}

            # توليد كود HTML للعرض 3D (Py3Dmol HTML Injection).
            view_html = render_protein_3d(
                pdb_data, bg_color=p['bg'], style_type=view_style,
                show_surface=show_surf, surface_opacity=surface_op,
                mutations=highlight if show_mutations else None,
                mut_color='#F44336' if prefix == 'm' else '#4CAF50',
                zoom_to_mutations=zoom_mutations, focus_mut=focus_mut, keep_hetatm=keep_hetatm
            )
            components.html(view_html, height=460)

            # مؤشرات إحصائية حيوية (Biological Metrics).
            total_res = sum(len([r for r in struct[0][c] if r.id[0] == ' ']) for c in get_all_chains(struct))
            c1, c2, c3 = st.columns(3)
            c1.metric("عدد سلاسل الببتيد", len(get_all_chains(struct)))
            c2.metric("إجمالي الأحماض الأمينية", total_res)
            c3.metric("السلسلة المختارة", selected_chain)

            # تحميل FASTA للتطبيقات الخارجية (مثل التنبؤ بمرض السرطان عبر برامج أخرى).
            fasta = sequence_to_fasta(struct, selected_chain, st.session_state.get(f"{prefix}_id", p['label']))
            if fasta:
                with st.expander(f"🧬 تصدير تسلسل FASTA - {p['label']}"):
                    st.download_button("⬇️ تحميل الملف التسلسلي", fasta, f"{st.session_state.get(f'{prefix}_id', 'protein')}_{selected_chain}.fasta", "text/plain", key=f"dl_f_{prefix}")

            # التحليل الهيكلي المكاني (Full Structural Distance Analysis).
            st.divider()
            if st.button(f"🔬 تشغيل التحليل الهيكلي الكامل - {p['label']}", key=f"analyze_btn_{prefix}", type="primary"):
                with st.spinner("جاري حساب المسافات الجزيئية وشبكة التفاعلات..."):
                    results = calculate_all_distances(f"{st.session_state.get(f'{prefix}_id')}_{selected_chain}", pdb_data, selected_chain, radius=search_radius, keep_hetatm=keep_hetatm)
                    st.session_state[f"{prefix}_results"] = results
            
            if st.session_state.get(f"{prefix}_results"):
                res = st.session_state[f"{prefix}_results"]
                with st.expander("📊 تفاصيل شبكة التفاعلات و الـ SASA"):
                    df = pd.DataFrame(res)
                    df_display = df.rename(columns={
                        'res_num': 'رقم الحمض', 'res_name': 'اسم الحمض', 'class': 'الخواص الكيميائية',
                        'min_dist': 'أقرب تفاعل (Å)', 'sasa': 'المساحة (SASA)'
                    })
                    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # 5. مرحلة المقارنة والتقرير الفني (Comparison & Report Phase).
    if structures.get('h') and structures.get('m') and h_chain and m_chain:
        st.divider()
        st.header("📋 المقارنة التحليلية المتقدمة")
        h_seq = get_protein_sequence(structures['h'], h_chain)
        m_seq = get_protein_sequence(structures['m'], m_chain)
        
        if h_seq and m_seq:
            # دمج بيانات الـ SASA لكل من السليم والمصاب للمقارنة المباشرة.
            h_sasa = calculate_sasa_map(structures['h'], h_chain)
            m_sasa = calculate_sasa_map(structures['m'], m_chain)

            df_h = pd.DataFrame(h_seq).rename(columns={'res_name': 'السليم (Healthy)'})
            df_m = pd.DataFrame(m_seq).rename(columns={'res_name': 'المصاب (Mutated)'})
            
            # دمج السلاسل (Merging/Joining): في حالة وجود فجوات (Gaps)، الـ Outer Join يضمن عدم ضياع أي موقع.
            df_comp = pd.merge(df_h, df_m, on='res_num', how='outer').sort_values('res_num').fillna('-')
            
            # حساب Delta SASA: التغير في مساحة السطح المتاح للمذيب. هو مفتاح معرفة ما إذا كانت الطفرة تسبب خللاً في طي البروتين.
            df_comp['SASA_H'] = df_comp['res_num'].map(lambda x: h_sasa.get(x, 0))
            df_comp['SASA_M'] = df_comp['res_num'].map(lambda x: m_sasa.get(x, 0))
            df_comp['SASA_Delta'] = df_comp['SASA_M'] - df_comp['SASA_H']
            
            # التنبؤ بالأثر العلمي (Impact Prediction).
            df_comp['Impact'] = df_comp.apply(lambda r: analyze_impact(r['السليم (Healthy)'], r['المصاب (Mutated)'], r['SASA_H'], r['SASA_M']), axis=1)
            df_comp['الحالة'] = df_comp.apply(lambda r: '🔴 طفرة مؤكدة' if r['السليم (Healthy)'] != r['المصاب (Mutated)'] else '🟢 حمض محافظ', axis=1)
            
            with st.expander("جدول المقارنة الفيزيائية والكيميائية (Physicochemical Comparison)"):
                df_disp = df_comp.rename(columns={
                    'res_num': 'موقع الحمض',
                    'SASA_H': 'SASA السليم',
                    'SASA_M': 'SASA المصاب',
                    'SASA_Delta': 'التغير ΔSASA',
                    'Impact': 'نوع التأثير العلمي المكتشف'
                })
                # تلوين صفوف الطفرات باللون الأحمر الداكن لجذب انتباه الباحث.
                st.dataframe(df_disp.style.apply(lambda r: ['background-color: #3e2723' if r['السليم (Healthy)'] != r['المصاب (Mutated)'] else ''] * len(r), axis=1), use_container_width=True, hide_index=True)

            # 6. مرحلة المحاذاة والتقرير الفني (Sequence Alignment & Technical Report).
            st.header(f"🧬 محاذاة السلاسل ({align_mode.capitalize()} Alignment)")
            h_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in h_seq])
            m_str = "".join([AA_3TO1.get(r['res_name'], 'X') for r in m_seq])
            
            aln_text, score, aln1, aln2 = get_alignment(h_str, m_str, align_mode)
            
            # حساب نسبة الهوية (Sequence Identity): إذا كانت أقل من 95%، الطفرات تعتبر كبيرة ومؤثرة جداً.
            matches = sum(a == b and a != '-' for a, b in zip(aln1, aln2))
            identity = (matches / len(aln1) * 100) if aln1 else 0
            
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("درجة المحاذاة (Score)", round(score, 1))
            sc2.metric("نسبة التطابق (Identity)", f"{identity:.1f}%")
            sc3.metric("فرق طول السلسلة", abs(len(h_str) - len(m_str)))
            st.code(aln_text, language='text')

            # ── توليد التقرير الفني النهائي (Automated Report Generation) ──
            st.divider()
            total_mutations = len(df_comp[df_comp['الحالة'] == '🔴 طفرة مؤكدة'])
            
            report_content = f"""
============================================================
              BIO-IMPACT ANALYZER - TECHNICAL REPORT
============================================================
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Project ID  : BIO-{st.session_state.get('m_id', 'UNKNOWN')}
------------------------------------------------------------
[1] EXECUTIVE SUMMARY
- Total Residues Analyzed : {len(df_comp)}
- Sequence Identity       : {identity:.2f}%
- Mutation Count          : {total_mutations}
- Status                  : {'Variance Detected' if identity < 95 else 'Highly Similar'}

[2] PROTEIN METADATA
HEALTHY REFERENCE: {st.session_state.get('h_id', 'N/A')} (Chain {h_chain}, {len(h_str)} AA)
MUTATED SAMPLE   : {st.session_state.get('m_id', 'N/A')} (Chain {m_chain}, {len(m_str)} AA)

[3] DETAILED MUTATION IMPACT TABLE
{'Pos'.ljust(6)} | {'Healthy'.ljust(8)} | {'S-H'.ljust(6)} | {'Mutated'.ljust(8)} | {'S-M'.ljust(6)} | {'Delta'.ljust(6)} | {'Impact'}
{'-' * 85}
"""
            # استخراج الطفرات فقط للتقرير.
            mutations = df_comp[df_comp['الحالة'] == '🔴 طفرة مؤكدة']
            for _, row in mutations.iterrows():
                report_content += f"{str(row['res_num']).ljust(6)} | {str(row['السليم (Healthy)']).ljust(8)} | {str(row['SASA_H']).ljust(6)} | {str(row['المصاب (Mutated)']).ljust(8)} | {str(row['SASA_M']).ljust(6)} | {str(round(row['SASA_Delta'], 2)).ljust(6)} | {row['Impact']}\n"

            report_content += f"""
[4] SEQUENCE ALIGNMENT MAP
{aln_text}
============================================================
                END OF PROFESSIONAL REPORT
============================================================
"""
            st.download_button(
                label="📥 تحميل التقرير العلمي المتكامل (PDF/Text)",
                data=report_content,
                file_name=f"BioReport_{st.session_state.get('m_id', 'BIO')}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )

# تشغيل التطبيق (App Execution entry point).
if __name__ == "__main__":
    main()