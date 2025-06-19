"use client";

import { useState, useEffect } from "react";
import { Save, Download, Upload } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { toast } from "@/hooks/use-toast";
import { EducationInfo, EmploymentHistory, MainInfo } from "@/types/client";
import { AxiosInstance } from "@/lib/axios-instance";
import ResumeTab from "@/components/settings/resume-tab";
import VaiTab from "@/components/settings/vai-tab";
import { encodeBase64, decodeBase64 } from "@/lib/base64";

export default function SettingsPage() {
  const [mainInfo, setMainInfo] = useState<MainInfo>({
    name: "",
    email: "",
    phone: "",
    location: "",
    linkedin: "",
  });
  const [employmentHistory, setEmploymentHistory] = useState<
    EmploymentHistory[]
  >([]);
  const [educationInfo, setEducationInfo] = useState<EducationInfo>({
    school: "",
    degree: "",
    from: "",
    to: "",
    location: "",
  });
  const [resumeWriterPrompt, setResumeWriterPrompt] = useState<string>("");
  const [coverLetterPrompt, setCoverLetterPrompt] = useState<string>("");
  const [resumeTemplatePath, setResumeTemplatePath] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  const [initialData, setInitialData] = useState<{
    mainInfo: MainInfo;
    employmentHistory: EmploymentHistory[];
    educationInfo: EducationInfo;
    resumeWriterPrompt: string;
    coverLetterPrompt: string;
    resumeTemplatePath: string;
  }>({
    mainInfo: {
      name: "",
      email: "",
      phone: "",
      location: "",
      linkedin: "",
    },
    employmentHistory: [],
    educationInfo: {
      school: "",
      degree: "",
      from: "",
      to: "",
      location: "",
    },
    resumeWriterPrompt: "",
    coverLetterPrompt: "",
    resumeTemplatePath: "",
  });

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const { data } = await AxiosInstance.get("/api/settings");

        // Format dates for employment history
        const initialEmploymentHistory = data.employmentHistory || [];

        // Format dates for education info
        const formattedEducationInfo = data.educationInfo
          ? {
              ...data.educationInfo,
              from: data.educationInfo.from,
              to: data.educationInfo.to,
            }
          : {
              school: "",
              degree: "",
              from: "",
              to: "",
              location: "",
            };

        const initialMainInfo = data.mainInfo || {
          name: "",
          email: "",
          phone: "",
          location: "",
          linkedin: "",
        };

        setMainInfo(initialMainInfo);
        setEmploymentHistory(initialEmploymentHistory);
        setEducationInfo(formattedEducationInfo);
        setResumeWriterPrompt(data.resumeWriterPrompt || "");
        setCoverLetterPrompt(data.coverLetterPrompt || "");
        setResumeTemplatePath(data.resumeTemplatePath || "");

        setInitialData({
          mainInfo: initialMainInfo,
          employmentHistory: initialEmploymentHistory,
          educationInfo: formattedEducationInfo,
          resumeWriterPrompt: data.resumeWriterPrompt || "",
          coverLetterPrompt: data.coverLetterPrompt || "",
          resumeTemplatePath: data.resumeTemplatePath || "",
        });
      } catch (error) {
        console.error("Error fetching settings:", error);
        toast({
          title: "Error",
          description: "Failed to load settings. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    fetchSettings();
  }, []);

  useEffect(() => {
    if (!initialData) return;

    const hasMainInfoChanges =
      JSON.stringify(mainInfo) !== JSON.stringify(initialData.mainInfo);
    const hasEmploymentChanges =
      JSON.stringify(employmentHistory) !==
      JSON.stringify(initialData.employmentHistory);
    const hasEducationChanges =
      JSON.stringify(educationInfo) !==
      JSON.stringify(initialData.educationInfo);
    const hasResumeWriterPromptChanges =
      resumeWriterPrompt !== initialData.resumeWriterPrompt;
    const hasCoverLetterPromptChanges =
      coverLetterPrompt !== initialData.coverLetterPrompt;
    const hasResumeTemplatePathChanges =
      resumeTemplatePath !== initialData.resumeTemplatePath;

    setHasChanges(
      hasMainInfoChanges ||
        hasEmploymentChanges ||
        hasEducationChanges ||
        hasResumeWriterPromptChanges ||
        hasCoverLetterPromptChanges ||
        hasResumeTemplatePathChanges
    );
  }, [
    mainInfo,
    employmentHistory,
    educationInfo,
    resumeWriterPrompt,
    coverLetterPrompt,
    resumeTemplatePath,
    initialData,
  ]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault(); // Prevent browser's default save dialog
        if (hasChanges) {
          handleSave();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hasChanges]); // Only re-run if hasChanges changes

  const handleMainInfoChange = (field: string, value: string) => {
    setMainInfo((prev) => ({ ...prev, [field]: value }));
  };

  const handleEmploymentHistoryChange = (
    index: number,
    field: string,
    value: string
  ) => {
    const newHistory = [...employmentHistory];
    if (!newHistory[index]) {
      newHistory[index] = {
        company: "",
        title: "",
        from: "",
        to: "",
        location: "",
        description: "",
        // projects: "",
      };
    }
    newHistory[index] = { ...newHistory[index], [field]: value };
    setEmploymentHistory(newHistory);
    setHasChanges(true);
  };

  const handleEducationInfoChange = (field: string, value: string) => {
    setEducationInfo((prev) => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await AxiosInstance.put("/api/settings", {
        mainInfo,
        employmentHistory: employmentHistory.map((item) => ({
          company: item.company,
          title: item.title,
          from: item.from,
          to: item.to,
          location: item.location,
          description: item.description,
          // projects: item.projects,
        })),
        educationInfo: {
          ...educationInfo,
          from: educationInfo.from,
          to: educationInfo.to,
        },
        resumeWriterPrompt,
        coverLetterPrompt,
        resumeTemplatePath,
      });

      setInitialData({
        mainInfo: { ...mainInfo },
        employmentHistory: [...employmentHistory],
        educationInfo: { ...educationInfo },
        resumeWriterPrompt,
        coverLetterPrompt,
        resumeTemplatePath,
      });
      setHasChanges(false);

      toast({
        title: "Success",
        description: "Your settings have been saved successfully.",
      });
    } catch (error) {
      console.error("Error saving settings:", error);
      toast({
        title: "Error",
        description: "Failed to save settings. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };
/*


*/
  const handleExport = () => {
    const exportData = {
      mainInfo,
      employmentHistory,
      educationInfo,
      resumeWriterPrompt: encodeBase64(resumeWriterPrompt),
      coverLetterPrompt: encodeBase64(coverLetterPrompt),
      resumeTemplatePath,
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "settings.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target?.result as string);

        // Validate the imported data structure
        if (!data.mainInfo || !data.employmentHistory || !data.educationInfo) {
          throw new Error("Invalid settings file format");
        }

        // Update state with imported data
        setMainInfo(data.mainInfo);
        setEmploymentHistory(
          data.employmentHistory.map((item: EmploymentHistory) => ({
            ...item,
            from: item.from,
            to: item.to,
          }))
        );
        setEducationInfo({
          ...data.educationInfo,
          from: data.educationInfo.from,
          to: data.educationInfo.to,
        });
        setResumeWriterPrompt(decodeBase64(data.resumeWriterPrompt || ""));
        setCoverLetterPrompt(decodeBase64(data.coverLetterPrompt || ""));
        setResumeTemplatePath(data.resumeTemplatePath || "");

        toast({
          title: "Success",
          description: "Settings imported successfully.",
        });
      } catch (error) {
        console.error("Error importing settings:", error);
        toast({
          title: "Error",
          description:
            "Failed to import settings. Please check the file format.",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
  };

  if (isLoading) {
    return (
      <div className="container mx-auto py-10">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative container mx-auto mt-4">
      <div className="absolute top-0 right-6 flex gap-2">
        {hasChanges && (
          <Button
            onClick={handleSave}
            variant="outline"
            className="bg-green-600 text-white hover:bg-green-700 hover:text-white"
            disabled={isSaving}
          >
            {isSaving ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-900"></div>
            ) : (
              <Save className="size-4" />
            )}
            {isSaving ? "Saving..." : "Save Changes"}
          </Button>
        )}
        <Button onClick={handleExport} variant="outline">
          <Download className="size-4" />
          Export
        </Button>
        <Button
          variant="outline"
          onClick={() => document.getElementById("import-input")?.click()}
        >
          <Upload className="size-4" />
          Import
        </Button>
        <input
          id="import-input"
          type="file"
          accept=".json"
          className="hidden"
          onChange={handleImport}
        />
      </div>
      <Tabs defaultValue="resume" className="space-y-4">
        <TabsList>
          <TabsTrigger value="resume">Resume</TabsTrigger>
          <TabsTrigger value="vai">Vai Settings</TabsTrigger>
          <TabsTrigger value="account">Account</TabsTrigger>
        </TabsList>

        <ResumeTab
          mainInfo={mainInfo}
          employmentHistory={employmentHistory}
          educationInfo={educationInfo}
          handleMainInfoChange={handleMainInfoChange}
          handleEmploymentHistoryChange={handleEmploymentHistoryChange}
          handleEducationInfoChange={handleEducationInfoChange}
        />

        <VaiTab
          resumeWriterPrompt={resumeWriterPrompt}
          coverLetterPrompt={coverLetterPrompt}
          resumeTemplatePath={resumeTemplatePath}
          setResumeWriterPrompt={setResumeWriterPrompt}
          setCoverLetterPrompt={setCoverLetterPrompt}
          setResumeTemplatePath={setResumeTemplatePath}
        />
      </Tabs>
    </div>
  );
}
