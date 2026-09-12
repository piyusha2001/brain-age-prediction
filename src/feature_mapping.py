from pathlib import Path
import re
import joblib


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS_PATH = (
    PROJECT_ROOT
    / "models"
    / "feature_cols.pkl"
)


# --------------------------------------------------
# 1. Special cases
#
# Features whose names should NOT be interpreted only
# using the generic parser.
# --------------------------------------------------

SPECIAL_CASES = {

    # --------------------------------------------------
# Global cortical measurements
# --------------------------------------------------

"LhMeanThick":
    "mean left cerebral cortical thickness",

"RhMeanThick":
    "mean right cerebral cortical thickness",

"LhWhiteSurfArea":
    "left cerebral white matter surface area",

"RhWhiteSurfArea":
    "right cerebral white matter surface area",


# --------------------------------------------------
# Intracranial / global brain volumes
# --------------------------------------------------

"TotalIntracranialVol":
    "total intracranial volume",

"LhCortWMVol":
    "left cerebral cortical white matter volume",

"RhCortWMVol":
    "right cerebral cortical white matter volume",

"CortWMVol":
    "total cerebral cortical white matter volume",

"SubCortGMVol":
    "subcortical gray matter volume",

"TotalGMVol":
    "total gray matter volume",

"MaskVolToeTIV":
    "brain mask volume to estimated intracranial volume ratio",


# --------------------------------------------------
# Surface reconstruction measures
# --------------------------------------------------

"LhSurfaceHolesVol":
    "left cortical surface holes",

"RhSurfaceHolesVol":
    "right cortical surface holes",

"SurfaceHolesVol":
    "total cortical surface holes",


# --------------------------------------------------
# Cerebellum / brainstem
# --------------------------------------------------

"LhCerebellumWMVol":
    "left cerebellar white matter volume",

"RhCerebellumWMVol":
    "right cerebellar white matter volume",

"BrainStemVol":
    "brainstem volume",


# --------------------------------------------------
# Choroid plexus
# --------------------------------------------------

"LhChoroidPVol":
    "left choroid plexus volume",

"RhChoroidPVol":
    "right choroid plexus volume",


# --------------------------------------------------
# Ventricular / other structures
# --------------------------------------------------

"FifthVentVol":
    "cerebral fifth ventricle volume",

"OpticChiasmVol":
    "optic chiasm volume",


# --------------------------------------------------
# Corpus callosum
# --------------------------------------------------

"CCPosteriorVol":
    "posterior corpus callosum volume",

"CCMidPosteriorVol":
    "mid-posterior corpus callosum volume",

"CCCentralVol":
    "central corpus callosum volume",

"CCMidAnteriorVol":
    "mid-anterior corpus callosum volume",

"CCAnteriorVol":
    "anterior corpus callosum volume",

    # Ventricles / CSF
    "ThirdVentVol": {
    "display_name": "Third ventricle volume",
    "amass_query": "cerebral third ventricle volume",
    "plain_definition": (
        "The volume of the third ventricle, a fluid-filled "
        "space near the center of the brain."
    )
    },

    "FourthVentVol":
        "cerebral fourth ventricle volume",

    "CSFVol":
        "cerebrospinal fluid volume",

    # White-matter abnormalities
    "WMHypointensitiesVol":
        "brain white matter hypointensity volume",

    "NonWMHypointensitiesVol":
        "brain non-white matter hypointensity volume",

    # Whole-brain measures
    "BrainSegVol":
        "brain segmentation volume",

    "BrainSegVolNotVent":
        "brain segmentation volume excluding ventricles",

    "BrainSegVolNotVentSurf":
        "brain segmentation volume excluding ventricles",

    "BrainSegVolToeTIV":
        "brain segmentation volume to estimated intracranial volume ratio",

    "eTIV":
        "estimated total intracranial volume",

    "EstimatedTotalIntraCranialVol":
        "estimated total intracranial volume",

    "TotalGrayVol":
        "total brain gray matter volume",

    "SubCortGrayVol":
        "subcortical gray matter volume",

    "SupraTentorialVol":
        "supratentorial brain volume",

    "SupraTentorialVolNotVent":
        "supratentorial brain volume excluding ventricles",

    "SupraTentorialVolNotVentVox":
        "supratentorial brain volume excluding ventricles",

    "MaskVol":
        "brain mask volume",
}

NON_AMASS_FEATURES = {
    "LhSurfaceHolesVol",
    "RhSurfaceHolesVol",
    "SurfaceHolesVol",
}
# --------------------------------------------------
# 2. Cortical regions
#
# These are common FreeSurfer-style anatomical names.
# --------------------------------------------------

CORTICAL_REGIONS = {

    "bankssts":
        "banks of superior temporal sulcus",

    "caudalanteriorcingulate":
        "caudal anterior cingulate cortex",

    "caudalmiddlefrontal":
        "caudal middle frontal cortex",

    "cuneus":
        "cuneus",

    "entorhinal":
        "entorhinal cortex",

    "fusiform":
        "fusiform gyrus",

    "inferiorparietal":
        "inferior parietal cortex",

    "inferiortemporal":
        "inferior temporal cortex",

    "isthmuscingulate":
        "isthmus cingulate cortex",

    "lateraloccipital":
        "lateral occipital cortex",

    "lateralorbitofrontal":
        "lateral orbitofrontal cortex",

    "lingual":
        "lingual gyrus",

    "medialorbitofrontal":
        "medial orbitofrontal cortex",

    "middletemporal":
        "middle temporal cortex",

    "parahippocampal":
        "parahippocampal cortex",

    "paracentral":
        "paracentral lobule",

    "parsopercularis":
        "pars opercularis",

    "parsorbitalis":
        "pars orbitalis",

    "parstriangularis":
        "pars triangularis",

    "pericalcarine":
        "pericalcarine cortex",

    "postcentral":
        "postcentral gyrus",

    "posteriorcingulate":
        "posterior cingulate cortex",

    "precentral":
        "precentral gyrus",

    "precuneus":
        "precuneus",

    "rostralanteriorcingulate":
        "rostral anterior cingulate cortex",

    "rostralmiddlefrontal":
        "rostral middle frontal cortex",

    "superiorfrontal":
        "superior frontal cortex",

    "superiorparietal":
        "superior parietal cortex",

    "superiortemporal":
        "superior temporal cortex",

    "supramarginal":
        "supramarginal gyrus",

    "frontalpole":
        "frontal pole",

    "temporalpole":
        "temporal pole",

    "transversetemporal":
        "transverse temporal cortex",

    "insula":
        "insula",
}


# --------------------------------------------------
# 3. Subcortical / structural names
# --------------------------------------------------

STRUCTURE_NAMES = {

    "Accumbens":
        "nucleus accumbens",

    "Amygdala":
        "amygdala",

    "Caudate":
        "caudate nucleus",

    "Hippocampus":
        "hippocampus",

    "Pallidum":
        "globus pallidus",

    "Putamen":
        "putamen",

    "Thalamus":
        "thalamus",

    "ThalamusProper":
        "thalamus",

    "VentralDC":
        "ventral diencephalon",

    "LatVent":
        "lateral ventricle",

    "InfLatVent":
        "inferior lateral ventricle",

    "ChoroidPlexus":
        "choroid plexus",

    "CerebellumCortex":
        "cerebellar cortex",

    "CerebellumWhiteMatter":
        "cerebellar white matter",

    "CorticalWhiteMatter":
        "cortical white matter",

    "Cortex":
        "cerebral cortex",

    "Vessel":
        "intracranial vessel",
}


# --------------------------------------------------
# 4. Measurement suffixes
# --------------------------------------------------

MEASUREMENT_SUFFIXES = {

    "Thick":
        "cortical thickness",

    "Vol":
        "volume",

    "Area":
        "surface area",

    "MeanCurv":
        "mean cortical curvature",

    "GausCurv":
        "Gaussian curvature",

    "FoldInd":
        "folding index",

    "CurvInd":
        "curvature index",
}


# --------------------------------------------------
# Helper: split generic CamelCase
# --------------------------------------------------

def split_camel_case(text: str) -> str:

    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text
    )

    text = text.replace("_", " ")

    return text.lower().strip()


# --------------------------------------------------
# Main parser
# --------------------------------------------------

def parse_feature_name(feature_name: str):
    """
    Convert a model feature name into a medical,
    literature-search-friendly term.

    Returns:
        medical_name
        mapping_method
    """

    # ----------------------------------------------
    # A. Exact special case
    # ----------------------------------------------

    if feature_name in SPECIAL_CASES:

        return (
            SPECIAL_CASES[feature_name],
            "special_case"
        )


    working = feature_name

    # ----------------------------------------------
    # B. Hemisphere
    # ----------------------------------------------

    hemisphere = None

    if working.startswith("Rh"):

        hemisphere = "right"
        working = working[2:]

    elif working.startswith("Lh"):

        hemisphere = "left"
        working = working[2:]


    # ----------------------------------------------
    # C. Measurement type
    # ----------------------------------------------

    measurement = None

    # longest suffix first
    suffixes = sorted(
        MEASUREMENT_SUFFIXES.keys(),
        key=len,
        reverse=True
    )

    for suffix in suffixes:

        if working.endswith(suffix):

            measurement = (
                MEASUREMENT_SUFFIXES[suffix]
            )

            working = working[:-len(suffix)]

            break


    # ----------------------------------------------
    # D. Cortical region
    # ----------------------------------------------

    region_key = working.lower()

    if region_key in CORTICAL_REGIONS:

        region = CORTICAL_REGIONS[region_key]

        parts = []

        if hemisphere:
            parts.append(hemisphere)

        parts.append(region)

        if measurement:
            parts.append(measurement)

        return (
            " ".join(parts),
            "cortical_parser"
        )


    # ----------------------------------------------
    # E. Subcortical / structural region
    # ----------------------------------------------

    if working in STRUCTURE_NAMES:

        region = STRUCTURE_NAMES[working]

        parts = []

        if hemisphere:
            parts.append(hemisphere)

        parts.append(region)

        if measurement:
            parts.append(measurement)

        return (
            " ".join(parts),
            "structure_parser"
        )


    # ----------------------------------------------
    # F. Generic fallback
    #
    # This ensures EVERY feature still receives
    # a readable search term.
    # ----------------------------------------------

    readable_region = split_camel_case(
        working
    )

    parts = []

    if hemisphere:
        parts.append(hemisphere)

    if readable_region:
        parts.append(readable_region)

    if measurement:
        parts.append(measurement)

    medical_name = " ".join(parts)

    return (
        medical_name,
        "fallback"
    )


# --------------------------------------------------
# Function used by evidence_pipeline.py
# --------------------------------------------------

def clean_feature_name(feature_name: str) -> str:
    """
    Return the Amass-friendly search term for a feature.
    """

    if feature_name in SPECIAL_CASES:

        value = SPECIAL_CASES[feature_name]

        # Rich metadata format
        if isinstance(value, dict):
            return value["amass_query"]

        # Old/simple string format
        return value

    medical_name, _ = parse_feature_name(
        feature_name
    )

    return medical_name

def is_amass_searchable(feature_name: str) -> bool:
    """
    Return False for model features that should not
    be used as clinical/literature search terms.
    """

    return feature_name not in NON_AMASS_FEATURES

# --------------------------------------------------
# Validate ALL model features
# --------------------------------------------------

def validate_all_features():

    feature_cols = joblib.load(
        FEATURE_COLS_PATH
    )

    print("\nFEATURE MAPPING VALIDATION")
    print("=" * 65)

    print(
        f"\nTotal features in model: "
        f"{len(feature_cols)}"
    )

    method_counts = {
        "special_case": 0,
        "cortical_parser": 0,
        "structure_parser": 0,
        "fallback": 0,
    }

    fallback_features = []

    for feature in feature_cols:

        medical_name, method = (
            parse_feature_name(feature)
        )

        method_counts[method] += 1

        if method == "fallback":

            fallback_features.append(
                (
                    feature,
                    medical_name
                )
            )


    print("\nMapping breakdown:")
    print(
        "  Special cases:",
        method_counts["special_case"]
    )

    print(
        "  Cortical parser:",
        method_counts["cortical_parser"]
    )

    print(
        "  Structure parser:",
        method_counts["structure_parser"]
    )

    print(
        "  Fallback:",
        method_counts["fallback"]
    )


    # ----------------------------------------------
    # Show anything needing manual review
    # ----------------------------------------------

    if fallback_features:

        print(
            "\nFEATURES NEEDING REVIEW"
        )

        print("-" * 65)

        for raw, mapped in fallback_features:

            print(
                f"{raw:40s} -> {mapped}"
            )

    else:

        print(
            "\nAll features mapped using "
            "known anatomical rules."
        )


    return fallback_features


# --------------------------------------------------
# Run validation directly
# --------------------------------------------------

if __name__ == "__main__":

    validate_all_features()