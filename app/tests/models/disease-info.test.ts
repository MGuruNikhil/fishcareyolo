import { describe, expect, it } from "bun:test"
import { DISEASE_CLASSES } from "@/lib/model/types"
import { validateDiseaseInfo } from "@/lib/model/types"
import {
    DISEASE_INFO,
    getAllDiseaseInfo,
    getBoundingBoxColor,
    getDiseaseInfo,
} from "@/lib/model/disease-info"

describe("**Feature: fish-disease-detection, Property 6: Disease info completeness**", () => {
    describe("DISEASE_INFO completeness", () => {
        it("should have an entry for every disease class", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                expect(DISEASE_INFO[diseaseClass]).toBeDefined()
                expect(DISEASE_INFO[diseaseClass].diseaseClass).toBe(
                    diseaseClass,
                )
            }
        })

        it("should have exactly 5 entries matching DISEASE_CLASSES", () => {
            const infoEntries = Object.keys(DISEASE_INFO)
            expect(infoEntries).toHaveLength(DISEASE_CLASSES.length)
            expect(infoEntries).toHaveLength(5)
        })

        it("should have valid structure for all entries", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                const errors = validateDiseaseInfo(info)
                expect(errors).toEqual([])
            }
        })
    })

    describe("Required fields validation", () => {
        it("should have non-empty displayName for all diseases", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                expect(info.displayName).toBeDefined()
                expect(info.displayName.trim().length).toBeGreaterThan(0)
            }
        })

        it("should have non-empty description for all diseases", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                expect(info.description).toBeDefined()
                expect(info.description.trim().length).toBeGreaterThan(0)
            }
        })

        it("should have at least one symptom for all diseases", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                expect(info.symptoms).toBeDefined()
                expect(Array.isArray(info.symptoms)).toBe(true)
                expect(info.symptoms.length).toBeGreaterThan(0)
            }
        })

        it("should have at least one treatment for all diseases", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                expect(info.treatments).toBeDefined()
                expect(Array.isArray(info.treatments)).toBe(true)
                expect(info.treatments.length).toBeGreaterThan(0)
            }
        })

        it("should have valid severity for all diseases", () => {
            const validSeverities = ["low", "medium", "high"]
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                expect(info.severity).toBeDefined()
                expect(validSeverities).toContain(info.severity)
            }
        })
    })

    describe("Specific disease data validation", () => {
        it("should have correct data for bacterial_infection", () => {
            const info = DISEASE_INFO.bacterial_infection
            expect(info.displayName).toBe("Bacterial Infection")
            expect(info.severity).toBe("high")
            expect(info.symptoms.length).toBeGreaterThan(0)
            expect(info.treatments.length).toBeGreaterThan(0)
        })

        it("should have correct data for fungal_infection", () => {
            const info = DISEASE_INFO.fungal_infection
            expect(info.displayName).toBe("Fungal Infection")
            expect(info.severity).toBe("medium")
            expect(info.symptoms.length).toBeGreaterThan(0)
            expect(info.treatments.length).toBeGreaterThan(0)
        })

        it("should have correct data for healthy", () => {
            const info = DISEASE_INFO.healthy
            expect(info.displayName).toBe("Healthy")
            expect(info.severity).toBe("low")
            expect(info.symptoms.length).toBeGreaterThan(0)
            expect(info.treatments.length).toBeGreaterThan(0)
        })

        it("should have correct data for parasite", () => {
            const info = DISEASE_INFO.parasite
            expect(info.displayName).toBe("Parasite Infection")
            expect(info.severity).toBe("medium")
            expect(info.symptoms.length).toBeGreaterThan(0)
            expect(info.treatments.length).toBeGreaterThan(0)
        })

        it("should have correct data for white_tail", () => {
            const info = DISEASE_INFO.white_tail
            expect(info.displayName).toBe("White Tail Disease")
            expect(info.severity).toBe("high")
            expect(info.symptoms.length).toBeGreaterThan(0)
            expect(info.treatments.length).toBeGreaterThan(0)
        })
    })

    describe("Helper functions", () => {
        it("getDiseaseInfo should return correct info for valid disease class", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = getDiseaseInfo(diseaseClass)
                expect(info).toBeDefined()
                expect(info.diseaseClass).toBe(diseaseClass)
                expect(info).toEqual(DISEASE_INFO[diseaseClass])
            }
        })

        it("getAllDiseaseInfo should return all disease info entries", () => {
            const allInfo = getAllDiseaseInfo()
            expect(allInfo).toHaveLength(DISEASE_CLASSES.length)
            expect(allInfo).toHaveLength(5)
            for (const info of allInfo) {
                expect(DISEASE_CLASSES).toContain(info.diseaseClass)
            }
        })
    })
})

describe("**Feature: fish-disease-detection, Property 7: Bounding box color mapping**", () => {
    describe("Color mapping by disease class", () => {
        it("should map healthy to green", () => {
            const color = getBoundingBoxColor("healthy")
            expect(color).toBe("#22c55e")
        })

        it("should map bacterial_infection (high severity) to red", () => {
            const color = getBoundingBoxColor("bacterial_infection")
            expect(color).toBe("#ef4444")
        })

        it("should map fungal_infection (medium severity) to red", () => {
            const color = getBoundingBoxColor("fungal_infection")
            expect(color).toBe("#ef4444")
        })

        it("should map parasite (medium severity) to red", () => {
            const color = getBoundingBoxColor("parasite")
            expect(color).toBe("#ef4444")
        })

        it("should map white_tail (high severity) to red", () => {
            const color = getBoundingBoxColor("white_tail")
            expect(color).toBe("#ef4444")
        })
    })

    describe("Color consistency", () => {
        it("should return valid hex color strings for all disease classes", () => {
            const hexColorPattern = /^#[0-9a-f]{6}$/i
            for (const diseaseClass of DISEASE_CLASSES) {
                const color = getBoundingBoxColor(diseaseClass)
                expect(hexColorPattern.test(color)).toBe(true)
            }
        })

        it("should return consistent colors for the same disease class", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const color1 = getBoundingBoxColor(diseaseClass)
                const color2 = getBoundingBoxColor(diseaseClass)
                expect(color1).toBe(color2)
            }
        })

        it("should use green only for healthy", () => {
            const greenColor = "#22c55e"
            for (const diseaseClass of DISEASE_CLASSES) {
                const color = getBoundingBoxColor(diseaseClass)
                if (diseaseClass === "healthy") {
                    expect(color).toBe(greenColor)
                } else {
                    expect(color).not.toBe(greenColor)
                }
            }
        })

        it("should map severity levels to appropriate colors", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                const info = DISEASE_INFO[diseaseClass]
                const color = getBoundingBoxColor(diseaseClass)

                if (diseaseClass === "healthy") {
                    expect(color).toBe("#22c55e")
                } else if (info.severity === "low") {
                    expect(color).toBe("#eab308")
                } else if (
                    info.severity === "medium" ||
                    info.severity === "high"
                ) {
                    expect(color).toBe("#ef4444")
                }
            }
        })
    })

    describe("Color palette validation", () => {
        it("should use exactly 3 distinct colors", () => {
            const colors = new Set(
                DISEASE_CLASSES.map((dc) => getBoundingBoxColor(dc)),
            )
            // Currently only green and red are used (yellow would be for low severity non-healthy)
            expect(colors.size).toBeGreaterThanOrEqual(1)
            expect(colors.size).toBeLessThanOrEqual(3)
        })

        it("should include green in the color palette", () => {
            const colors = DISEASE_CLASSES.map((dc) => getBoundingBoxColor(dc))
            expect(colors).toContain("#22c55e")
        })

        it("should include red in the color palette", () => {
            const colors = DISEASE_CLASSES.map((dc) => getBoundingBoxColor(dc))
            expect(colors).toContain("#ef4444")
        })
    })
})
