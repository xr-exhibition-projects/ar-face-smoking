from app.helpers.typing_helper import LayoutDictTypes

# Widgets in Effect Params tab are created from this Layout
EFFECT_PARAMS_LAYOUT_DATA: LayoutDictTypes = {
    'Aging Effect': {
        'AgingStage1StartSlider': {
            'level': 1,
            'label': 'Stage 1 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Starting intensity (0-100%) for aging effect in Stage 1 (from "УПОРОТЬСЯ" to "СЛЕЗТЬ" button).'
        },
        'AgingStage1EndSlider': {
            'level': 1,
            'label': 'Stage 1 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Ending intensity (0-100%) for aging effect in Stage 1 (from "УПОРОТЬСЯ" to "СЛЕЗТЬ" button).'
        },
        'AgingStage2StartSlider': {
            'level': 1,
            'label': 'Stage 2 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Starting intensity (0-100%) for aging effect in Stage 2 (from "СЛЕЗТЬ" to "НЕВОЗМОЖНО" message).'
        },
        'AgingStage2EndSlider': {
            'level': 1,
            'label': 'Stage 2 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Ending intensity (0-100%) for aging effect in Stage 2 (from "СЛЕЗТЬ" to "НЕВОЗМОЖНО" message).'
        },
        'AgingStage3StartSlider': {
            'level': 1,
            'label': 'Stage 3 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Starting intensity (0-100%) for aging effect in Stage 3 (from "НЕВОЗМОЖНО" to "СМЕРТЬ НЕИЗБЕЖНА" screen).'
        },
        'AgingStage3EndSlider': {
            'level': 1,
            'label': 'Stage 3 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Ending intensity (0-100%) for aging effect in Stage 3 (from "НЕВОЗМОЖНО" to "СМЕРТЬ НЕИЗБЕЖНА" screen).'
        },
    },
    'Zombie Effect': {
        'ZombieStage1StartSlider': {
            'level': 1,
            'label': 'Stage 1 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Starting intensity (0-100%) for zombie effect in Stage 1 (from "УПОРОТЬСЯ" to "СЛЕЗТЬ" button).'
        },
        'ZombieStage1EndSlider': {
            'level': 1,
            'label': 'Stage 1 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Ending intensity (0-100%) for zombie effect in Stage 1 (from "УПОРОТЬСЯ" to "СЛЕЗТЬ" button).'
        },
        'ZombieStage2StartSlider': {
            'level': 1,
            'label': 'Stage 2 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '0',
            'step': 1,
            'help': 'Starting intensity (0-100%) for zombie effect in Stage 2 (from "СЛЕЗТЬ" to "НЕВОЗМОЖНО" message).'
        },
        'ZombieStage2EndSlider': {
            'level': 1,
            'label': 'Stage 2 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 1,
            'help': 'Ending intensity (0-100%) for zombie effect in Stage 2 (from "СЛЕЗТЬ" to "НЕВОЗМОЖНО" message).'
        },
        'ZombieStage3StartSlider': {
            'level': 1,
            'label': 'Stage 3 Start Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '80',
            'step': 1,
            'help': 'Starting intensity (0-100%) for zombie effect in Stage 3 (from "НЕВОЗМОЖНО" to "СМЕРТЬ НЕИЗБЕЖНА" screen).'
        },
        'ZombieStage3EndSlider': {
            'level': 1,
            'label': 'Stage 3 End Intensity',
            'min_value': '0',
            'max_value': '100',
            'default': '100',
            'step': 1,
            'help': 'Ending intensity (0-100%) for zombie effect in Stage 3 (from "НЕВОЗМОЖНО" to "СМЕРТЬ НЕИЗБЕЖНА" screen).'
        },
    },
    'Stage Timings': {
        'Stage1DurationSlider': {
            'level': 1,
            'label': 'Stage 1 Duration (ms)',
            'min_value': '100',
            'max_value': '30000',
            'default': '4000',
            'step': 100,
            'help': 'Duration in milliseconds for Stage 1 (from "УПОРОТЬСЯ" to "СЛЕЗТЬ" button).'
        },
        'Stage2DurationSlider': {
            'level': 1,
            'label': 'Stage 2 Duration (ms)',
            'min_value': '100',
            'max_value': '30000',
            'default': '2000',
            'step': 100,
            'help': 'Duration in milliseconds for Stage 2 (from "СЛЕЗТЬ" to "НЕВОЗМОЖНО" message).'
        },
        'Stage3DurationSlider': {
            'level': 1,
            'label': 'Stage 3 Duration (ms)',
            'min_value': '100',
            'max_value': '30000',
            'default': '5000',
            'step': 100,
            'help': 'Duration in milliseconds for Stage 3 (from "НЕВОЗМОЖНО" to "СМЕРТЬ НЕИЗБЕЖНА" screen).'
        },
    },
}

