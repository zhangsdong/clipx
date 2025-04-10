[33mcommit 8c251641aea4555bcfe596c5d45a34d9545c3046[m[33m ([m[1;36mHEAD -> [m[1;32mmain[m[33m)[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 13:43:54 2025 +0800

    feat(cli): make -o/--output parameter optional
    
    Make output parameter optional with default naming [input_file]_remove.png
    in the current directory when omitted. Update help text and validation logic
    to reflect this change.

[33mcommit 8c2dad02d6a9a59fb288d369c873a31e8cc3d7d9[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 13:05:30 2025 +0800

    refactor: optimize logging

[33mcommit 3f56c074c47fc8dbd9c90e74e28ec4d88ad60235[m[33m ([m[1;31morigin/main[m[33m, [m[1;31morigin/HEAD[m[33m)[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 11:16:09 2025 +0800

    feat: added logging.py configuration to optimize log printing

[33mcommit f4d72b1b8282d356f1574782d550865aee6fdc6c[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 10:33:44 2025 +0800

    update model file address

[33mcommit ae6c8be40062dde783a4a12c57fc7552b5897c06[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 10:06:26 2025 +0800

    refactor(cascadepsp): optimize module structure, move implementation code from __init__.py
    - Create cascadepsp/model.py file, move CascadePSPModel class implementation
    - Simplify __init__.py, keep only necessary imports and exports
    - Replace print statements with standard logging
    - Improve exception handling

[33mcommit fcc68a446aee014166307dcdc0cc78800d2f9e79[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 09:59:55 2025 +0800

    refactor(cascadepsp): simplify model download logic

[33mcommit 6d86d2503c8dd7cb6cff6cca60b0fd3c7c516844[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Thu Apr 10 09:51:32 2025 +0800

    refactor(u2net): simplify model download logic
    
    - Reduce code complexity in U2Net download module
    - Consolidate helper functions into a single verification function
    - Minimize progress output to essential checkpoints (25% intervals)
    - Streamline error handling and file verification flow
    - Remove redundant logging statements

[33mcommit 91162e1dcb5f288ccadacecaa19ed8551f183caa[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Wed Apr 9 13:50:50 2025 +0800

    feat: improve model handling, add timing metrics and error recovery
    
    - Create shared BaseModel class for consistent model interfaces
    - Implement automatic GPU/CPU detection and selection
    - Add detailed performance timing for each processing stage
    - Enhance error handling in CascadePSP with graceful fallback
    - Fix type conversion issues in model processing pipeline
    - Add version display option (-v/--version) to CLI
    - Improve logging with detailed time metrics for performance analysis

[33mcommit 3fa964ff2ffc18b75e62e487bb7aa873c5e5a4bc[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Mon Apr 7 15:10:05 2025 +0800

    fix(cli): fix -v parameter error

[33mcommit 01fd477293c1f25062d4881c03fbaea0ed8c9ff2[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Mon Apr 7 14:34:25 2025 +0800

    fix(cli): fix parameter error

[33mcommit d54068d123327a98a1551d301a22e3cffe923633[m
Author: zhangsdong <zhangsdong0402@gmail.com>
Date:   Mon Apr 7 11:32:49 2025 +0800

    feat: integrate CascadePSP model for mask refinement
    
    - Port segmentation_refinement component from CascadePSP project
    - Fix module import paths to adapt to clipx project structure
    - Enhance class and method docstrings
    - Add GPU support and fast processing mode
    - Implement command line interface for CascadePSP model
    
    Based on original code: https://github.com/hkchengrex/CascadePSP

[33mcommit 9b7c6ddc588ee7801b3a91fe36b5a6ef7fe891cf[m
Author: zhangsdong <99520922+zhangsdong@users.noreply.github.com>
Date:   Tue Apr 1 16:36:13 2025 +0800

    Initial commit
