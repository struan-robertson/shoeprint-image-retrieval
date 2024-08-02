((python-base-mode
  . ((eglot-workspace-configuration
      . (:pylsp
	 (:plugins
	  (:jedi
	   (:environment ".venv/")
	   :ruff (:line_length 88
			   :target_version "py310"
			   :extendSelect ["ALL"]
			   :extendIgnore ["ANN" ;; Type hinting
					  "FIX" ;; fixme should be handled by emacs not ruff
					  "TD" ;; same with todo
					  "C90" ;; mccabe complexity
					  "DJ" ;; Django
					  "EXE" ;; flake8-executable
					  "T20" ;; allow-print
					  ]
			   :format ["I"]
			   :executable "/usr/bin/ruff"
			   )))))
     (indent-tabs-mode . nil)))
 (nil
  . ((python-shell-interpreter . "~/Development/Doctorate/shoeprint-image-retrieval/.venv/bin/python")
     (python-shell-virtualenv-root . "~/Development/Doctorate/shoeprint-image-retrieval/.venv/")
     (python-shell-process-environment . '("HSA_OVERRIDE_GFX_VERSION=10.3.0")))))
