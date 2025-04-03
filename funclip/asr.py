from dataclasses import dataclass
from typing import List
from collections import defaultdict

from funasr import AutoModel
import numpy as np

from utils.time_utils import Duration, Timestamp, SentenceSRT

def asr(audio_binary:np.ndarray, use_main_speaker=True) -> List[SentenceSRT]:
    
    funasr_model = AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", # Non-ar asr
        vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", # support any length
        punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", # split to sentences
        spk_model="damo/speech_campplus_sv_zh-cn_16k-common" if use_main_speaker else None, # speaker recongination
        disable_update=True,
        device='cuda:0'
    )

    rec_result = funasr_model.generate(
        audio_binary, 
        return_spk_res=use_main_speaker, 
        sentence_timestamp=True, 
        return_raw_text=True, 
        # is_final=True, 
        # hotword="",
        # output_dir=None,
        # cache={}
    )

    if use_main_speaker:
        speakers = defaultdict(list)
        for sent in rec_result[0]['sentence_info']:
            speakers[sent['spk']].append(SentenceSRT.from_dict(sent))
        
        max_spk = max(speakers, key=lambda x: len(speakers[x]))
        return speakers[max_spk]
    else:
        return [SentenceSRT.from_dict(sent) for sent in rec_result[0]['sentence_info']]


