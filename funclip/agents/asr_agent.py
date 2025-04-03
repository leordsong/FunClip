from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import gc

from funasr import AutoModel
import numpy as np
import torch

from agents.base_agent import BaseAgent
from logger import logger
from utils.time_utils import SentenceSRT


class ASRAgent(BaseAgent):
    """
    ASR Agent for automatic speech recognition.
    """

    def __init__(self, use_main_speaker=False) -> None:
        super().__init__()
        self.use_main_speaker = use_main_speaker
        self._model = None  # Placeholder for the ASR model

    def open(self) -> None:
        """Open the ASR agent."""
        logger.info("Loading ASR model...")
        self._model = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", # Non-ar asr
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", # support any length
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", # split to sentences
            spk_model="damo/speech_campplus_sv_zh-cn_16k-common" if self.use_main_speaker else None, # speaker recongination
            disable_update=True,
            device='cuda:0'
        )
        logger.info("ASR model loaded successfully.")

    def close(self) -> None:
        """Close the ASR agent."""
        # Unload the ASR model here
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, audio_np:np.ndarray) -> List[SentenceSRT]:
        """Call the ASR agent."""
        # Implement the ASR functionality here
        rec_result = self._model.generate(
            audio_np, 
            return_spk_res=self.use_main_speaker, 
            sentence_timestamp=True, 
            return_raw_text=True, 
            # is_final=True, 
            # hotword="",
            # output_dir=None,
            # cache={}
        )

        if self.use_main_speaker:
            speakers = defaultdict(list)
            for sent in rec_result[0]['sentence_info']:
                speakers[sent['spk']].append(SentenceSRT.from_dict(sent))
            
            max_spk = max(speakers, key=lambda x: len(speakers[x]))
            return speakers[max_spk]
        else:
            return [SentenceSRT.from_dict(sent) for sent in rec_result[0]['sentence_info']]