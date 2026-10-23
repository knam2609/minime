// Written by scripts/build_showcase.py from the committed reports: do not edit.
window.MINIME = {
 "compute": {
  "kaggle_gpu_hours": 34.0,
  "laptop_hours": 6.9,
  "params": 49262400,
  "runtime": "8h39m",
  "tokens": 904396800
 },
 "final": {
  "chosen": "2026-10-19, by the owner, after reading the voice samples",
  "final_run": "m10-adapt-c",
  "rows": [
   {
    "better": "lower",
    "difference_test": -0.103758,
    "difference_val": -0.113482,
    "final_test": 3.623481,
    "final_val": 3.663631,
    "flagship_test": 3.72724,
    "flagship_val": 3.777113,
    "metric": "personal CE",
    "primary": true
   },
   {
    "better": "lower",
    "difference_test": 0.01125,
    "difference_val": 0.011007,
    "final_test": 3.400644,
    "final_val": 3.394652,
    "flagship_test": 3.389394,
    "flagship_val": 3.383645,
    "metric": "general-weighted CE",
    "primary": true
   },
   {
    "better": "lower",
    "difference_test": 0.040893,
    "difference_val": 0.035787,
    "final_test": 0.689241,
    "final_val": 0.698216,
    "flagship_test": 0.648348,
    "flagship_val": 0.662429,
    "metric": "impersonation AUC",
    "primary": false
   },
   {
    "better": "higher",
    "difference_test": 0.068359,
    "difference_val": 0.115234,
    "final_test": 0.951172,
    "final_val": 0.921875,
    "flagship_test": 0.882812,
    "flagship_val": 0.806641,
    "metric": "attributed to the owner",
    "primary": false
   },
   {
    "better": "higher",
    "difference_test": 0.090909,
    "difference_val": 0.230769,
    "final_test": 1.0,
    "final_val": 1.0,
    "flagship_test": 0.909091,
    "flagship_val": 0.769231,
    "metric": "Delta: nearest is the owner",
    "primary": false
   },
   {
    "better": "lower",
    "difference_test": 0.013911,
    "difference_val": 0.001531,
    "final_test": 0.066373,
    "final_val": 0.046057,
    "flagship_test": 0.052462,
    "flagship_val": 0.044526,
    "metric": "function-word JSD to real",
    "primary": false
   },
   {
    "better": "higher",
    "difference_test": -0.043478,
    "difference_val": -0.089431,
    "final_test": 0.434783,
    "final_val": 0.455285,
    "flagship_test": 0.478261,
    "flagship_val": 0.544715,
    "metric": "code syntax validity",
    "primary": false
   },
   {
    "better": "higher",
    "difference_test": 0.018955,
    "difference_val": 0.018955,
    "final_test": 0.758209,
    "final_val": 0.758209,
    "flagship_test": 0.739254,
    "flagship_val": 0.739254,
    "metric": "BLiMP accuracy",
    "primary": false
   }
  ]
 },
 "frontier": {
  "test": {
   "context_length": 512,
   "front": [
    "flagship",
    "adaptation",
    "E1 25/75",
    "E1 50/50"
   ],
   "groups": {
    "E1 25/75": {
     "label": "E1 25/75",
     "mean": {
      "general": 3.922958,
      "personal": 3.580446
     },
     "runs": {
      "e1-m25-p1s1337": {
       "general": 3.915579,
       "personal": 3.594746
      },
      "e1-m25-p1s1338": {
       "general": 3.928034,
       "personal": 3.575061
      },
      "e1-m25-p1s1339": {
       "general": 3.92187,
       "personal": 3.586545
      },
      "e1-m25-p1s1340": {
       "general": 3.926349,
       "personal": 3.565432
      }
     },
     "size": "17M",
     "what": "P1, then adapted on 25% chat, 75% general"
    },
    "E1 50/50": {
     "label": "E1 50/50",
     "mean": {
      "general": 4.003593,
      "personal": 3.524689
     },
     "runs": {
      "e1-m50-p1s1337": {
       "general": 3.993366,
       "personal": 3.533809
      },
      "e1-m50-p1s1338": {
       "general": 4.004832,
       "personal": 3.524334
      },
      "e1-m50-p1s1339": {
       "general": 4.005109,
       "personal": 3.514947
      },
      "e1-m50-p1s1340": {
       "general": 4.011066,
       "personal": 3.525665
      }
     },
     "size": "17M",
     "what": "P1, then adapted on 50% chat, 50% general"
    },
    "P1": {
     "label": "P1",
     "mean": {
      "general": 3.726135,
      "personal": 7.359896
     },
     "runs": {
      "p1-s1337": {
       "general": 3.722,
       "personal": 7.483993
      },
      "p1-s1338": {
       "general": 3.731252,
       "personal": 7.375159
      },
      "p1-s1339": {
       "general": 3.725459,
       "personal": 7.378296
      },
      "p1-s1340": {
       "general": 3.725828,
       "personal": 7.202137
      }
     },
     "size": "17M",
     "what": "general data only"
    },
    "P4": {
     "label": "P4",
     "mean": {
      "general": 3.729484,
      "personal": 3.793272
     },
     "runs": {
      "p4-s1337": {
       "general": 3.72799,
       "personal": 3.788214
      },
      "p4-s1338": {
       "general": 3.725344,
       "personal": 3.80097
      },
      "p4-s1339": {
       "general": 3.731627,
       "personal": 3.785595
      },
      "p4-s1340": {
       "general": 3.732974,
       "personal": 3.798308
      }
     },
     "size": "17M",
     "what": "the owner's chat mixed in from step 0"
    },
    "adaptation": {
     "label": "Flagship + C",
     "mean": {
      "general": 3.460508,
      "personal": 3.650645
     },
     "runs": {
      "m10-adapt-c": {
       "general": 3.460508,
       "personal": 3.650645
      }
     },
     "size": "49M",
     "what": "the flagship, adapted by candidate C: the final model"
    },
    "flagship": {
     "label": "Flagship",
     "mean": {
      "general": 3.446945,
      "personal": 3.791011
     },
     "runs": {
      "flagship-s1337": {
       "general": 3.446945,
       "personal": 3.791011
      }
     },
     "size": "49M",
     "what": "the owner's chat mixed in from step 0"
    }
   },
   "split": "test"
  },
  "val": {
   "context_length": 512,
   "front": [
    "flagship",
    "adaptation",
    "E1 25/75",
    "E1 50/50"
   ],
   "groups": {
    "E1 25/75": {
     "label": "E1 25/75",
     "mean": {
      "general": 3.916547,
      "personal": 3.572183
     },
     "runs": {
      "e1-m25-p1s1337": {
       "general": 3.909469,
       "personal": 3.581539
      },
      "e1-m25-p1s1338": {
       "general": 3.920913,
       "personal": 3.561407
      },
      "e1-m25-p1s1339": {
       "general": 3.915217,
       "personal": 3.578698
      },
      "e1-m25-p1s1340": {
       "general": 3.920591,
       "personal": 3.567087
      }
     },
     "size": "17M",
     "what": "P1, then adapted on 25% chat, 75% general"
    },
    "E1 50/50": {
     "label": "E1 50/50",
     "mean": {
      "general": 3.99875,
      "personal": 3.490929
     },
     "runs": {
      "e1-m50-p1s1337": {
       "general": 3.988582,
       "personal": 3.4905
      },
      "e1-m50-p1s1338": {
       "general": 3.999528,
       "personal": 3.493671
      },
      "e1-m50-p1s1339": {
       "general": 3.999326,
       "personal": 3.493566
      },
      "e1-m50-p1s1340": {
       "general": 4.007565,
       "personal": 3.485981
      }
     },
     "size": "17M",
     "what": "P1, then adapted on 50% chat, 50% general"
    },
    "P1": {
     "label": "P1",
     "mean": {
      "general": 3.718372,
      "personal": 7.379661
     },
     "runs": {
      "p1-s1337": {
       "general": 3.714866,
       "personal": 7.512007
      },
      "p1-s1338": {
       "general": 3.72311,
       "personal": 7.360851
      },
      "p1-s1339": {
       "general": 3.716692,
       "personal": 7.491994
      },
      "p1-s1340": {
       "general": 3.718821,
       "personal": 7.153794
      }
     },
     "size": "17M",
     "what": "general data only"
    },
    "P4": {
     "label": "P4",
     "mean": {
      "general": 3.722024,
      "personal": 3.809318
     },
     "runs": {
      "p4-s1337": {
       "general": 3.720335,
       "personal": 3.81197
      },
      "p4-s1338": {
       "general": 3.717864,
       "personal": 3.793414
      },
      "p4-s1339": {
       "general": 3.725762,
       "personal": 3.811574
      },
      "p4-s1340": {
       "general": 3.724133,
       "personal": 3.820313
      }
     },
     "size": "17M",
     "what": "the owner's chat mixed in from step 0"
    },
    "adaptation": {
     "label": "Flagship + C",
     "mean": {
      "general": 3.451933,
      "personal": 3.684429
     },
     "runs": {
      "m10-adapt-c": {
       "general": 3.451933,
       "personal": 3.684429
      }
     },
     "size": "49M",
     "what": "the flagship, adapted by candidate C: the final model"
    },
    "flagship": {
     "label": "Flagship",
     "mean": {
      "general": 3.438256,
      "personal": 3.837797
     },
     "runs": {
      "flagship-s1337": {
       "general": 3.438256,
       "personal": 3.837797
      }
     },
     "size": "49M",
     "what": "the owner's chat mixed in from step 0"
    }
   },
   "split": "val"
  }
 },
 "memorisation": {
  "groups": [
   {
    "gap": 0.0,
    "group": "P1",
    "n": 4,
    "overlap_mean": 0.855957,
    "overlap_p95": 4.0,
    "train_rate": 0.0,
    "val_rate": 0.0
   },
   {
    "gap": -0.000714,
    "group": "P4",
    "n": 4,
    "overlap_mean": 1.788574,
    "overlap_p95": 5.75,
    "train_rate": 0.006071,
    "val_rate": 0.006786
   },
   {
    "gap": -0.005357,
    "group": "E1 50/50",
    "n": 4,
    "overlap_mean": 2.360352,
    "overlap_p95": 6.75,
    "train_rate": 0.0075,
    "val_rate": 0.012857
   },
   {
    "gap": -0.001429,
    "group": "E1 100/0",
    "n": 4,
    "overlap_mean": 2.780273,
    "overlap_p95": 7.5,
    "train_rate": 0.010714,
    "val_rate": 0.012143
   },
   {
    "gap": 0.002857,
    "group": "flagship",
    "n": 1,
    "overlap_mean": 1.701172,
    "overlap_p95": 6.0,
    "train_rate": 0.004286,
    "val_rate": 0.001429
   },
   {
    "gap": 0.002857,
    "group": "flagship + C",
    "n": 1,
    "overlap_mean": 1.763672,
    "overlap_p95": 6.0,
    "train_rate": 0.007143,
    "val_rate": 0.004286
   }
  ],
  "real": {
   "max": 17,
   "overlap_mean": 2.744141,
   "overlap_p95": 7.45
  },
  "sensitive": false,
  "threshold": 7.45
 },
 "recipe": {
  "adapt_tokens": 1048576,
  "candidate": "C",
  "candidates": 3,
  "change_val": {
   "general": 0.011007,
   "personal": -0.113482
  },
  "chat_share_adapt": 25,
  "chat_share_pretrain": 0.115942,
  "chat_tokens": 1048576,
  "chosen": "2026-10-19, by the owner, after reading the voice samples",
  "general_cap": 0.02,
  "general_parts": {
   "code": 1723,
   "dialogue": 3446,
   "web": 12061
  },
  "lr": 4e-05,
  "personal_gain": 0.1
 },
 "study": {
  "e1": {
   "100": {
    "general": {
     "difference": 1.937615,
     "half_width": 0.110253,
     "verdict": "higher than the control"
    },
    "personal": {
     "difference": -3.15943,
     "half_width": 0.167064,
     "verdict": "lower than the control"
    }
   },
   "25": {
    "general": {
     "difference": 0.049934,
     "half_width": 0.007876,
     "verdict": "higher than the control"
    },
    "personal": {
     "difference": -3.38938,
     "half_width": 0.156115,
     "verdict": "lower than the control"
    }
   },
   "50": {
    "general": {
     "difference": 0.132137,
     "half_width": 0.019796,
     "verdict": "higher than the control"
    },
    "personal": {
     "difference": -3.470634,
     "half_width": 0.179379,
     "verdict": "lower than the control"
    }
   }
  },
  "e3": {
   "difference": 0.109862,
   "half_width": 0.016363,
   "verdict": "higher than the baseline"
  },
  "e4": {
   "difference": -0.042547,
   "half_width": 0.016363,
   "verdict": "lower than the baseline"
  },
  "floor": {
   "half_width": 0.016363,
   "metric": "final_val_loss",
   "n": 4
  },
  "p4": {
   "e1-50": {
    "general": {
     "difference": -0.276726,
     "half_width": 0.023118,
     "verdict": "lower than e1-50"
    },
    "personal": {
     "difference": 0.318389,
     "half_width": 0.032092,
     "verdict": "higher than e1-50"
    }
   },
   "p1": {
    "general": {
     "difference": 0.003651,
     "half_width": 0.0136,
     "verdict": "unresolved"
    },
    "personal": {
     "difference": -3.570344,
     "half_width": 0.445416,
     "verdict": "lower than p1"
    }
   }
  }
 }
};
