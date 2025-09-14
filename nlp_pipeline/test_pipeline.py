"""
Comprehensive test suite for the Text-to-CAD NLP Pipeline
Demonstrates advanced capabilities across all complexity levels
"""

import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp_pipeline.core.pipeline import (
    ArchitecturalNLPPipeline,
    PipelineConfig,
    ProcessingMode
)


class NLPPipelineTestSuite:
    """Comprehensive test suite for NLP pipeline"""

    def __init__(self):
        """Initialize test suite with different pipeline configurations"""
        self.pipelines = {
            "basic": ArchitecturalNLPPipeline(
                PipelineConfig(mode=ProcessingMode.BASIC)
            ),
            "residential": ArchitecturalNLPPipeline(
                PipelineConfig(mode=ProcessingMode.RESIDENTIAL)
            ),
            "commercial": ArchitecturalNLPPipeline(
                PipelineConfig(mode=ProcessingMode.COMMERCIAL)
            ),
            "complex": ArchitecturalNLPPipeline(
                PipelineConfig(mode=ProcessingMode.COMPLEX)
            )
        }

        self.test_cases = self._load_test_cases()

    def _load_test_cases(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load comprehensive test cases for each complexity level"""
        return {
            "L0_basic": [
                {
                    "input": "Create a wall 5 meters long and 3 meters high",
                    "expected_entities": ["wall"],
                    "expected_dimensions": {"length": 5.0, "height": 3.0},
                    "expected_intent": "CREATE"
                },
                {
                    "input": "Add a concrete column with diameter of 500mm and height 4m",
                    "expected_entities": ["column"],
                    "expected_materials": ["concrete"],
                    "expected_dimensions": {"diameter": 0.5, "height": 4.0}
                },
                {
                    "input": "Build a rectangular slab 10ft by 15ft",
                    "expected_entities": ["slab"],
                    "expected_dimensions": {"length": 4.572, "width": 3.048}  # Converted to meters
                }
            ],
            "L1_residential": [
                {
                    "input": "Design a two-story house with 3 bedrooms, 2 bathrooms, " +
                            "living room 5m x 6m, and kitchen 4m x 4m",
                    "expected_entities": ["house", "bedroom", "bathroom", "living room", "kitchen"],
                    "expected_quantities": {"bedroom": 3, "bathroom": 2},
                    "expected_areas": {"living_room": 30.0, "kitchen": 16.0}
                },
                {
                    "input": "Create a master bedroom 4m x 5m with ensuite bathroom " +
                            "and walk-in closet, north-facing windows",
                    "expected_entities": ["bedroom", "bathroom", "closet", "window"],
                    "expected_relationships": ["ensuite", "walk-in", "north-facing"],
                    "expected_area": 20.0
                },
                {
                    "input": "Add a pitched roof with 30 degree slope, clay tiles, " +
                            "and 600mm overhang on all sides",
                    "expected_entities": ["roof"],
                    "expected_materials": ["clay tiles"],
                    "expected_properties": {"slope": 30, "overhang": 0.6}
                }
            ],
            "L2_commercial": [
                {
                    "input": "Design a 5-story office building with open floor plans, " +
                            "2 elevators, emergency stairs at both ends, " +
                            "curtain wall facade with 40% glazing ratio",
                    "expected_entities": ["office building", "elevator", "stairs", "curtain wall"],
                    "expected_quantities": {"floors": 5, "elevators": 2},
                    "expected_properties": {"glazing_ratio": 0.4}
                },
                {
                    "input": "Create a retail space 20m x 30m with 5m ceiling height, " +
                            "loading dock at rear, customer entrance at front, " +
                            "and mezzanine level for storage",
                    "expected_entities": ["retail space", "loading dock", "entrance", "mezzanine"],
                    "expected_dimensions": {"length": 20, "width": 30, "height": 5},
                    "expected_area": 600.0
                },
                {
                    "input": "Configure HVAC system for 1000 sqm office space with " +
                            "VAV boxes, 25 tons cooling capacity, and BMS integration",
                    "expected_entities": ["HVAC system", "VAV boxes", "BMS"],
                    "expected_properties": {"area": 1000, "cooling_capacity": 25}
                }
            ],
            "L3_complex": [
                {
                    "input": "Design an iconic museum with parametric facade inspired by " +
                            "ocean waves, cantilevered galleries extending 15m, " +
                            "central atrium with ETFE roof spanning 40m diameter",
                    "expected_entities": ["museum", "facade", "galleries", "atrium", "roof"],
                    "expected_properties": {
                        "facade_type": "parametric",
                        "cantilever": 15.0,
                        "atrium_span": 40.0,
                        "roof_material": "ETFE"
                    }
                },
                {
                    "input": "Create a sustainable skyscraper with double-skin facade, " +
                            "sky gardens every 5 floors, wind turbines at 200m height, " +
                            "and diagrid structural system with 6-story modules",
                    "expected_entities": ["skyscraper", "facade", "sky gardens", "wind turbines", "diagrid"],
                    "expected_properties": {
                        "facade_type": "double-skin",
                        "garden_frequency": 5,
                        "turbine_height": 200,
                        "module_height": 6
                    }
                },
                {
                    "input": "Model the Oculus transportation hub with elliptical plan " +
                            "111m x 35m, ribbed steel structure opening to 20m at apex, " +
                            "and retractable skylight system",
                    "expected_entities": ["transportation hub", "steel structure", "skylight"],
                    "expected_dimensions": {"length": 111, "width": 35, "apex_opening": 20},
                    "expected_properties": {"plan_shape": "elliptical", "skylight_type": "retractable"}
                }
            ],
            "multi_intent": [
                {
                    "input": "Create a conference room 8m x 10m then add sliding glass " +
                            "partitions on the east wall and modify ceiling height to 3.5m",
                    "expected_intents": ["CREATE", "CREATE", "MODIFY"],
                    "expected_entities": ["conference room", "partitions", "ceiling"]
                },
                {
                    "input": "Analyze the structural load on the main beam and optimize " +
                            "the column spacing for better load distribution",
                    "expected_intents": ["ANALYZE", "OPTIMIZE"],
                    "expected_entities": ["beam", "column"]
                }
            ],
            "constraint_extraction": [
                {
                    "input": "Design office space complying with IBC 2021, minimum ceiling " +
                            "height 2.7m, maximum occupancy 50 people, and fire rating 2 hours",
                    "expected_constraints": {
                        "building_code": "IBC 2021",
                        "min_ceiling_height": 2.7,
                        "max_occupancy": 50,
                        "fire_rating": "2 hours"
                    }
                },
                {
                    "input": "Create accessible bathroom with ADA compliance, door width " +
                            "minimum 32 inches, grab bars at 33-36 inches height",
                    "expected_constraints": {
                        "compliance": "ADA",
                        "min_door_width": 0.8128,  # 32 inches in meters
                        "grab_bar_height_range": [0.838, 0.914]  # 33-36 inches in meters
                    }
                }
            ],
            "material_specifications": [
                {
                    "input": "Build walls with reinforced concrete C30/37, steel beams " +
                            "S355, and triple-glazed windows with U-value 0.8 W/m²K",
                    "expected_materials": {
                        "concrete": {"grade": "C30/37", "type": "reinforced"},
                        "steel": {"grade": "S355"},
                        "glass": {"type": "triple-glazed", "u_value": 0.8}
                    }
                }
            ],
            "spatial_relationships": [
                {
                    "input": "Place kitchen adjacent to dining room, master bedroom above " +
                            "living room, and garage connected to mudroom",
                    "expected_relationships": [
                        {"source": "kitchen", "target": "dining room", "type": "adjacent"},
                        {"source": "master bedroom", "target": "living room", "type": "above"},
                        {"source": "garage", "target": "mudroom", "type": "connected"}
                    ]
                }
            ],
            "complex_dimensions": [
                {
                    "input": "Create room 12'-6\" x 10'-3\" with ceiling height varying " +
                            "from 8ft to 12ft, and wall thickness 200mm ± 10mm",
                    "expected_dimensions": {
                        "length": 3.81,  # 12'-6" in meters
                        "width": 3.124,  # 10'-3" in meters
                        "min_height": 2.438,  # 8ft in meters
                        "max_height": 3.658,  # 12ft in meters
                        "wall_thickness": {"value": 0.2, "tolerance": 0.01}
                    }
                }
            ]
        }

    async def run_all_tests(self):
        """Run all test cases and generate report"""
        print("=" * 80)
        print("TEXT-TO-CAD NLP PIPELINE TEST SUITE")
        print("=" * 80)
        print()

        results = {}
        for category, test_cases in self.test_cases.items():
            print(f"\n{'='*60}")
            print(f"Testing Category: {category.upper()}")
            print(f"{'='*60}")

            category_results = []
            for i, test_case in enumerate(test_cases, 1):
                result = await self._run_single_test(category, test_case, i)
                category_results.append(result)

            results[category] = category_results

        # Generate summary report
        self._generate_report(results)

    async def _run_single_test(
        self,
        category: str,
        test_case: Dict[str, Any],
        test_num: int
    ) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\nTest {test_num}:")
        print(f"Input: \"{test_case['input'][:100]}...\""
              if len(test_case['input']) > 100 else f"Input: \"{test_case['input']}\"")

        # Select appropriate pipeline based on category
        if category.startswith("L0"):
            pipeline = self.pipelines["basic"]
        elif category.startswith("L1"):
            pipeline = self.pipelines["residential"]
        elif category.startswith("L2"):
            pipeline = self.pipelines["commercial"]
        elif category.startswith("L3"):
            pipeline = self.pipelines["complex"]
        else:
            pipeline = self.pipelines["residential"]  # Default

        try:
            # Process input
            result = await pipeline.process_async(test_case['input'])

            # Validate results
            validation = self._validate_result(test_case, result)

            # Print summary
            print(f"  ✓ Processing Time: {result.processing_time:.3f}s")
            print(f"  ✓ Confidence: {result.confidence_scores.get('overall', 0):.2%}")
            print(f"  ✓ Entities Found: {len(result.entities)}")
            print(f"  ✓ Intent: {result.intent}")

            if validation['passed']:
                print(f"  ✅ Test PASSED")
            else:
                print(f"  ❌ Test FAILED: {validation['reason']}")

            return {
                "test_case": test_case,
                "result": result,
                "validation": validation,
                "passed": validation['passed']
            }

        except Exception as e:
            print(f"  ❌ Test FAILED with error: {e}")
            return {
                "test_case": test_case,
                "error": str(e),
                "passed": False
            }

    def _validate_result(
        self,
        test_case: Dict[str, Any],
        result: Any
    ) -> Dict[str, Any]:
        """Validate test results against expected values"""
        validation = {"passed": True, "reason": "", "details": {}}

        # Check expected entities
        if "expected_entities" in test_case:
            found_entities = [e.name.lower() for e in result.entities]
            for expected in test_case["expected_entities"]:
                if expected.lower() not in str(found_entities).lower():
                    validation["passed"] = False
                    validation["reason"] = f"Missing expected entity: {expected}"

        # Check expected intent
        if "expected_intent" in test_case:
            if result.intent != test_case["expected_intent"]:
                validation["passed"] = False
                validation["reason"] = f"Wrong intent: expected {test_case['expected_intent']}, got {result.intent}"

        # Check dimensions
        if "expected_dimensions" in test_case:
            for dim_name, expected_value in test_case["expected_dimensions"].items():
                found = False
                for dim_list in result.dimensions.values():
                    if isinstance(dim_list, list):
                        for dim in dim_list:
                            if hasattr(dim, 'value'):
                                # Allow 10% tolerance for dimension matching
                                if abs(dim.value - expected_value) / expected_value < 0.1:
                                    found = True
                                    break

                if not found:
                    validation["passed"] = False
                    validation["reason"] = f"Missing dimension: {dim_name}={expected_value}"

        return validation

    def _generate_report(self, results: Dict[str, List[Dict[str, Any]]]):
        """Generate test summary report"""
        print("\n" + "=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)

        total_tests = 0
        passed_tests = 0

        for category, category_results in results.items():
            category_passed = sum(1 for r in category_results if r.get("passed", False))
            category_total = len(category_results)

            total_tests += category_total
            passed_tests += category_passed

            print(f"\n{category}: {category_passed}/{category_total} passed " +
                  f"({category_passed/category_total*100:.1f}%)")

        print("\n" + "-" * 40)
        print(f"OVERALL: {passed_tests}/{total_tests} tests passed " +
              f"({passed_tests/total_tests*100:.1f}%)")

        # Performance metrics
        all_times = []
        all_confidences = []

        for category_results in results.values():
            for result in category_results:
                if "result" in result:
                    all_times.append(result["result"].processing_time)
                    all_confidences.append(
                        result["result"].confidence_scores.get("overall", 0)
                    )

        if all_times:
            print("\n" + "-" * 40)
            print("PERFORMANCE METRICS:")
            print(f"  Average Processing Time: {sum(all_times)/len(all_times):.3f}s")
            print(f"  Min Processing Time: {min(all_times):.3f}s")
            print(f"  Max Processing Time: {max(all_times):.3f}s")

        if all_confidences:
            print(f"  Average Confidence: {sum(all_confidences)/len(all_confidences):.2%}")
            print(f"  Min Confidence: {min(all_confidences):.2%}")
            print(f"  Max Confidence: {max(all_confidences):.2%}")

        print("\n" + "=" * 80)

    async def run_interactive_demo(self):
        """Run interactive demo for manual testing"""
        print("\n" + "=" * 80)
        print("INTERACTIVE NLP PIPELINE DEMO")
        print("=" * 80)
        print("\nEnter architectural descriptions to test the pipeline.")
        print("Type 'exit' to quit, 'examples' for sample inputs.")
        print("-" * 80)

        examples = [
            "Create a modern house with 4 bedrooms and open plan kitchen",
            "Build a 20-story office tower with glass curtain wall facade",
            "Design sustainable school with solar panels and green roof",
            "Add french doors connecting living room to garden terrace",
            "Modify wall height to 3.5 meters and change material to exposed brick",
            "Analyze structural integrity of the cantilever beam",
            "Generate floor plan for 150 sqm apartment with 3 bedrooms"
        ]

        while True:
            try:
                user_input = input("\n> ").strip()

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'examples':
                    print("\nExample inputs:")
                    for i, example in enumerate(examples, 1):
                        print(f"  {i}. {example}")
                    continue
                elif not user_input:
                    continue

                # Process input
                print("\nProcessing...")
                pipeline = self.pipelines["residential"]  # Use residential as default
                result = await pipeline.process_async(user_input)

                # Display results
                print("\n" + "-" * 60)
                print("RESULTS:")
                print("-" * 60)

                print(f"\n📍 Intent: {result.intent}")
                print(f"   Confidence: {result.confidence_scores.get('overall', 0):.2%}")

                if result.entities:
                    print(f"\n🏗️ Entities ({len(result.entities)}):")
                    for entity in result.entities[:5]:  # Show first 5
                        print(f"   • {entity.name} ({entity.type.value}) - {entity.confidence:.2%}")

                if result.dimensions.get("linear"):
                    print(f"\n📏 Dimensions:")
                    for dim in result.dimensions["linear"][:5]:  # Show first 5
                        print(f"   • {dim.type.value}: {dim.value:.2f} {dim.unit}")

                if result.materials:
                    print(f"\n🎨 Materials:")
                    for material in result.materials[:5]:  # Show first 5
                        print(f"   • {material.name} - {material.type.value}")

                if result.spatial_relationships:
                    print(f"\n🔗 Spatial Relationships:")
                    for rel in result.spatial_relationships[:5]:  # Show first 5
                        print(f"   • {rel.source_element} {rel.relationship_type.value} {rel.target_element}")

                if result.warnings:
                    print(f"\n⚠️ Warnings:")
                    for warning in result.warnings:
                        print(f"   • {warning}")

                if result.suggestions:
                    print(f"\n💡 Suggestions:")
                    for suggestion in result.suggestions:
                        print(f"   • {suggestion}")

                print(f"\n⏱️ Processing time: {result.processing_time:.3f} seconds")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


async def main():
    """Main entry point"""
    test_suite = NLPPipelineTestSuite()

    print("Select mode:")
    print("1. Run all automated tests")
    print("2. Interactive demo")
    print("3. Both")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        await test_suite.run_all_tests()
    elif choice == "2":
        await test_suite.run_interactive_demo()
    elif choice == "3":
        await test_suite.run_all_tests()
        await test_suite.run_interactive_demo()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())